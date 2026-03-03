"""
eac_synth/pipeline.py
End-to-end EAC-Synth pipeline orchestrator.

Chains all four stages into a single callable:
    Stage 1: Rare entity mining     (idf_filter + ner_extractor)
    Stage 2: Subgraph construction  (graph_builder)
    Stage 3: Longest-chain extract  (chain_extractor)
    Stage 4: Verbalization + RCDC   (verbalizer + rcdc)

Usage::

    from eac_synth.pipeline import EACSynthPipeline
    from eac_synth.synthesis.verbalizer import make_openai_teacher

    teacher = make_openai_teacher(model="gpt-4o-mini")
    pipeline = EACSynthPipeline.from_teacher(teacher, domain="medical")
    dataset = pipeline.run(corpus_docs, entity_to_docs, N=4200)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from eac_synth.entity.idf_filter import compute_corpus_idf, select_rare_entities
from eac_synth.graph.graph_builder import GraphBuilder, make_llm_edge_scorer
from eac_synth.graph.chain_extractor import LongestChainExtractor, make_llm_chain_scorer
from eac_synth.synthesis.verbalizer import Verbalizer
from eac_synth.synthesis.rcdc import RCDCFilter


class EACSynthPipeline:
    """Four-stage EAC-Synth corpus synthesis pipeline.

    Args:
        graph_builder:    configured GraphBuilder instance (Stage 2).
        chain_extractor:  configured LongestChainExtractor instance (Stage 3).
        verbalizer:       configured Verbalizer instance (Stage 4).
        rcdc_filter:      configured RCDCFilter instance (Stage 4).
        kappa:            rarity ceiling for IDF filtering (default 10).
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        chain_extractor: LongestChainExtractor,
        verbalizer: Verbalizer,
        rcdc_filter: RCDCFilter,
        kappa: int = 10,
    ):
        self.builder = graph_builder
        self.extractor = chain_extractor
        self.verbalizer = verbalizer
        self.rcdc = rcdc_filter
        self.kappa = kappa


    @classmethod
    def from_teacher(
        cls,
        teacher_fn: Callable,
        domain: str,
        kg_client: Optional[Any] = None,
        llm_discoverer: Optional[Callable] = None,
        kappa: int = 10,
        alpha: float = 0.5,
        depth: int = 3,
        edge_score_thresh: float = 0.5,
        chain_top_k: int = 5,
        chain_min_hops: int = 2,
        chain_score_thresh: float = 0.7,
        rcdc_gamma: float = 0.5,
        rcdc_theta: float = 0.75,
        verbalize_temperature: float = 0.85,
        verbalize_top_p: float = 0.90,
    ) -> "EACSynthPipeline":
        """Build a pipeline from a teacher callable and a domain label.

        Args:
            teacher_fn: callable(system, user, temperature, top_p) -> dict.
                Wraps the LLM API (see verbalizer.make_openai_teacher).
            domain: domain label, e.g. "medical", "legal", "scientific".
            kg_client: KG neighbor client; if None, a stub that returns []
                is used (LLM-discovery branch will be used exclusively).
            llm_discoverer: if None, defaults to a wrapper around teacher_fn.
            All other args: hyper-parameters with paper defaults.
        """
        from eac_synth.graph.graph_builder import StubKGClient

        if kg_client is None:
            kg_client = StubKGClient({})

        # Edge scorer: wraps teacher_fn to return Eq.(5) score
        def _edge_scorer(src: str, rel: str, dst: str) -> float:
            prompt_sys = "You are a domain expert."
            prompt_usr = (
                f"Evaluate the triple: ({src}, {rel}, {dst})\n"
                "Return JSON: {\"p_rel\": float, \"p_dom\": float}"
            )
            try:
                r = teacher_fn(prompt_sys, prompt_usr, temperature=0.0, top_p=1.0)
                p_rel = float(r.get("p_rel", 0.5))
                p_dom = float(r.get("p_dom", 0.5))
                lam = 0.4
                return lam * p_rel + (1 - lam) * p_dom
            except Exception:
                return 0.0

        # LLM discoverer: prompts teacher to suggest new neighbors
        if llm_discoverer is None:
            def llm_discoverer(entity: str, passages: List[str]) -> List:
                context = "\n".join(passages[:3])
                prompt_sys = "You are a domain expert."
                prompt_usr = (
                    f"Given the entity '{entity}' and context:\n{context}\n\n"
                    "List up to 5 domain-relevant related entities and their "
                    "relationships as JSON: "
                    "{\"neighbors\": [{\"entity\": str, \"relation\": str}]}"
                )
                try:
                    r = teacher_fn(prompt_sys, prompt_usr, temperature=0.0, top_p=1.0)
                    return [(n["entity"], n["relation"]) for n in r.get("neighbors", [])]
                except Exception:
                    return []

        # Chain scorer
        def _chain_scorer(path, domain_label):
            from eac_synth.graph.chain_extractor import (
                _SYSTEM_PROMPT, _USER_TEMPLATE, format_chain_for_prompt
            )
            chain_str = format_chain_for_prompt(path)
            user_msg = _USER_TEMPLATE.format(domain=domain_label, chain=chain_str)
            try:
                return teacher_fn(_SYSTEM_PROMPT, user_msg, temperature=0.0, top_p=1.0)
            except Exception:
                return {"valid": False, "score": 0.0, "reason": "error"}

        builder = GraphBuilder(
            kg_client=kg_client,
            llm_discoverer=llm_discoverer,
            edge_scorer=_edge_scorer,
            alpha=alpha,
            depth=depth,
            score_thresh=edge_score_thresh,
        )
        extractor = LongestChainExtractor(
            llm_scorer=_chain_scorer,
            domain=domain,
            top_k=chain_top_k,
            min_hops=chain_min_hops,
            score_thresh=chain_score_thresh,
        )
        verb = Verbalizer(
            teacher_fn=teacher_fn,
            domain=domain,
            temperature=verbalize_temperature,
            top_p=verbalize_top_p,
        )
        rcdc = RCDCFilter(gamma=rcdc_gamma, theta_div=rcdc_theta)

        return cls(builder, extractor, verb, rcdc, kappa=kappa)


    def run(
        self,
        corpus_docs: List[List[str]],
        entity_to_docs: Dict[str, set],
        N: int,
        ontology_check: Optional[Callable[[str], bool]] = None,
        passage_retriever: Optional[Callable[[str], List[str]]] = None,
        verbose: bool = True,
    ) -> List[dict]:
        """Run the full four-stage pipeline and return the synthesised corpus.

        Args:
            corpus_docs: list of token-lists (one per document).
            entity_to_docs: entity -> set of doc IDs index.
            N: total corpus document count.
            ontology_check: callable(entity) -> bool for Stage 1 validation.
                Defaults to accepting all entities.
            passage_retriever: callable(entity) -> List[str] returning the
                top-5 TF-IDF passages for graph expansion.
                Defaults to returning empty list (triggers LLM-only expansion).
            verbose: print progress.

        Returns:
            List of accepted sample dicts with keys:
                question, steps, answer, path.
        """
        if ontology_check is None:
            ontology_check = lambda e: True
        if passage_retriever is None:
            passage_retriever = lambda e: []

        idf = compute_corpus_idf(entity_to_docs, N)
        rare_entities = select_rare_entities(idf, N, ontology_check, self.kappa)
        if verbose:
            print(f"Stage 1: {len(rare_entities)} rare seed entities identified.")

        corpus: List[dict] = []
        skipped = 0

        for idx, (entity, idf_score) in enumerate(rare_entities):
            if verbose and (idx + 1) % 100 == 0:
                pct = (idx + 1) / len(rare_entities) * 100
                print(f"  [{idx+1}/{len(rare_entities)}  {pct:.0f}%]  "
                      f"accepted={len(corpus)}  skipped={skipped}")

            passages = passage_retriever(entity)
            subgraph = self.builder.build(entity, passages)

            path = self.extractor.extract(subgraph)
            if path is None:
                skipped += 1
                continue

            sample = self.verbalizer.verbalize(path)
            if sample is None:
                skipped += 1
                continue

            if self.rcdc.accept(sample):
                corpus.append(sample)

        if verbose:
            rejection_rate = 1 - len(corpus) / max(1, len(rare_entities) - skipped)
            print(f"\nDone. Accepted: {len(corpus)}  "
                  f"RCDC rejection rate: {rejection_rate:.1%}  "
                  f"Subgraphs skipped: {skipped}")

        return corpus


    @staticmethod
    def save(corpus: List[dict], path: str) -> None:
        """Save synthesised corpus as JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(corpus)} samples to {path}")

    @staticmethod
    def load(path: str) -> List[dict]:
        """Load a previously saved corpus."""
        return json.load(open(path))
