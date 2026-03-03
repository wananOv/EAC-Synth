"""
tests/test_core.py
Unit tests for EAC-Synth core modules.

Run with:  pytest tests/test_core.py -v
"""

import math
import pytest

from eac_synth.entity.idf_filter import compute_corpus_idf, select_rare_entities
from eac_synth.graph.graph_builder import Edge, Subgraph, StubKGClient, GraphBuilder
from eac_synth.graph.chain_extractor import (
    LongestChainExtractor, format_chain_for_prompt,
)
from eac_synth.synthesis.rcdc import RCDCFilter, jaccard_3g, rel_sim, rel_seq




class TestIDFFilter:

    def _make_index(self):
        # entity_to_docs: e -> set of doc IDs
        return {
            "LHON":         {0, 1, 2},   # doc_freq = 3, IDF = log(10/4)
            "MT-ND4":       {0},          # doc_freq = 1, IDF = log(10/2)
            "common_gene":  {0,1,2,3,4,5,6,7,8,9},  # too common
        }

    def test_idf_values(self):
        index = self._make_index()
        idf = compute_corpus_idf(index, N=10)
        assert idf["LHON"]   == pytest.approx(math.log(10 / 4), rel=1e-6)
        assert idf["MT-ND4"] == pytest.approx(math.log(10 / 2), rel=1e-6)

    def test_rare_entity_selection_kappa10(self):
        index = self._make_index()
        idf = compute_corpus_idf(index, N=10)
        # kappa=10 -> tau_rare = log(10/10) = 0; everything above 0 qualifies
        rare = select_rare_entities(idf, N=10, ontology_check=lambda e: True, kappa=10)
        names = [e for e, _ in rare]
        # All three qualify (IDF > 0), sorted descending
        assert "MT-ND4" in names  # highest IDF
        assert "LHON" in names

    def test_ontology_gate(self):
        index = {"good": {0}, "bad": {1}}
        idf = compute_corpus_idf(index, N=100)
        rare = select_rare_entities(
            idf, N=100,
            ontology_check=lambda e: e == "good",
            kappa=10,
        )
        assert len(rare) == 1
        assert rare[0][0] == "good"

    def test_returns_sorted_descending(self):
        index = {"a": {0}, "b": {0, 1}, "c": {0, 1, 2}}
        idf = compute_corpus_idf(index, N=100)
        rare = select_rare_entities(idf, N=100, ontology_check=lambda e: True, kappa=50)
        scores = [s for _, s in rare]
        assert scores == sorted(scores, reverse=True)


class TestGraphBuilder:

    def _make_builder(self, score=0.8):
        kg = StubKGClient({
            "LHON": [("MT-ND4", "encoded_by"), ("optic", "affects")],
            "MT-ND4": [("vision", "impairs")],
        })
        return GraphBuilder(
            kg_client=kg,
            llm_discoverer=lambda e, p: [],
            edge_scorer=lambda s, r, d: score,
            alpha=1.0,   # always KG branch for determinism
            depth=2,
            score_thresh=0.5,
        )

    def test_builds_subgraph(self):
        builder = self._make_builder()
        g = builder.build("LHON", passages=[])
        assert g.seed == "LHON"
        assert "LHON" in g.nodes
        assert len(g.edges) > 0

    def test_acyclicity(self):
        builder = self._make_builder()
        g = builder.build("LHON", passages=[])
        node_set = set(g.nodes)
        assert len(node_set) == len(g.nodes), "Duplicate nodes found (cycle)"

    def test_score_threshold_filters(self):
        kg = StubKGClient({"seed": [("neighbor", "rel")]})
        builder = GraphBuilder(
            kg_client=kg,
            llm_discoverer=lambda e, p: [],
            edge_scorer=lambda s, r, d: 0.3,   # below threshold
            alpha=1.0,
            depth=1,
            score_thresh=0.5,
        )
        g = builder.build("seed", [])
        assert len(g.edges) == 0  # filtered out


def _make_path(*triples):
    """Helper: build a List[Edge] from (src, rel, dst) tuples."""
    return [Edge(src=s, relation=r, dst=d) for s, r, d in triples]


class TestChainExtractor:

    def _make_subgraph(self):
        g = Subgraph(seed="A")
        edges = [
            Edge("A", "r1", "B"), Edge("B", "r2", "C"),
            Edge("C", "r3", "D"), Edge("D", "r4", "E"),
        ]
        g.edges = edges
        g.nodes = ["A", "B", "C", "D", "E"]
        return g

    def _always_valid_scorer(self, path, domain):
        return {"valid": True, "score": 0.9, "reason": "ok"}

    def test_extracts_longest(self):
        ext = LongestChainExtractor(
            llm_scorer=self._always_valid_scorer,
            domain="medical",
            top_k=5,
            min_hops=2,
            score_thresh=0.7,
        )
        g = self._make_subgraph()
        path = ext.extract(g)
        assert path is not None
        assert len(path) >= 2

    def test_returns_none_if_min_hops_not_met(self):
        ext = LongestChainExtractor(
            llm_scorer=self._always_valid_scorer,
            domain="medical",
            min_hops=10,   # impossible for 4-hop graph
            score_thresh=0.7,
        )
        g = self._make_subgraph()
        assert ext.extract(g) is None

    def test_returns_none_if_score_below_threshold(self):
        def low_scorer(path, domain):
            return {"valid": False, "score": 0.3, "reason": "nope"}
        ext = LongestChainExtractor(
            llm_scorer=low_scorer,
            domain="medical",
            min_hops=1,
            score_thresh=0.7,
        )
        g = self._make_subgraph()
        assert ext.extract(g) is None

    def test_format_chain_for_prompt(self):
        path = _make_path(("A", "causes", "B"), ("B", "inhibits", "C"))
        s = format_chain_for_prompt(path)
        assert "causes" in s
        assert "inhibits" in s
        assert "A" in s and "C" in s


class TestRCDC:

    def _sample(self, question: str, relations: list) -> dict:
        return {
            "question": question,
            "path": [{"relation": r} for r in relations],
        }

    def test_accepts_diverse_samples(self):
        filt = RCDCFilter(gamma=0.5, theta_div=0.75)
        s1 = self._sample("What causes optic neuropathy?", ["causes", "leads_to"])
        s2 = self._sample("Which gene encodes the frataxin protein?", ["encodes", "localises_to"])
        assert filt.accept(s1) is True
        assert filt.accept(s2) is True
        assert filt.pool_size == 2

    def test_rejects_near_duplicate(self):
        filt = RCDCFilter(gamma=0.5, theta_div=0.75)
        s1 = self._sample("What causes optic neuropathy in LHON?", ["causes", "leads_to"])
        s2 = self._sample("What causes optic neuropathy in LHON?", ["causes", "leads_to"])
        filt.accept(s1)
        assert filt.accept(s2) is False  # identical -> rejected

    def test_pool_grows_on_accept(self):
        filt = RCDCFilter()
        for i in range(5):
            filt.accept(self._sample(f"Unique question number {i} about different topic", [f"rel{i}"]))
        assert filt.pool_size == 5

    def test_jaccard_3g_identical(self):
        assert jaccard_3g("hello world", "hello world") == pytest.approx(1.0)

    def test_jaccard_3g_disjoint(self):
        # Two strings with no overlapping 3-grams
        assert jaccard_3g("abc", "xyz") == pytest.approx(0.0)

    def test_rel_sim_identical(self):
        assert rel_sim(("a", "b"), ("a", "b")) == pytest.approx(1.0)

    def test_rel_sim_different(self):
        assert rel_sim(("a", "b"), ("c", "d")) < 0.5

    def test_rel_seq_extraction(self):
        sample = {"path": [{"relation": "causes"}, {"relation": "inhibits"}], "question": ""}
        assert rel_seq(sample) == ("causes", "inhibits")

    def test_batch_filter(self):
        filt = RCDCFilter(theta_div=0.75)
        samples = [
            self._sample("Question about mitochondria pathway alpha", ["causes", "encodes"]),
            self._sample("Question about mitochondria pathway alpha", ["causes", "encodes"]),  # dup
            self._sample("Entirely different topic about legal contracts", ["governs", "restricts"]),
        ]
        accepted = filt.batch_filter(samples)
        assert len(accepted) == 2  # second is rejected as duplicate

    def test_reset(self):
        filt = RCDCFilter()
        filt.accept(self._sample("q", ["r"]))
        assert filt.pool_size == 1
        filt.reset()
        assert filt.pool_size == 0
