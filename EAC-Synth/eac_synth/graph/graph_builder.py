"""
eac_synth/graph/graph_builder.py
Entity-Relation Subgraph Construction -- Stage 2 of EAC-Synth.

Implements Equations (4)-(6) from the paper:
    Eq.(4)  Iterative neighbourhood expansion (KG vs LLM branch)
    Eq.(5)  LLM-assisted edge scoring with lambda-weighted combination
    Eq.(6)  Enriched edge annotation with temporal and context tags

After construction, subgraphs contain an average of 14.7 nodes and
17.4 validated edges (paper, Table 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple




@dataclass
class Edge:
    """A directed, annotated edge in the entity-relation subgraph.

    Corresponds to the enriched relation r^natural in Eq.(6):
        r^natural = <e_subj, p, e_obj, t_temporal, c_ctx>
    """
    src: str
    relation: str
    dst: str
    score: float = 0.0          # combined score from Eq.(5)
    temporal: Optional[str] = None   # temporal tag extracted from passage
    ctx: Optional[str] = None        # context tag extracted from passage

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "relation": self.relation,
            "dst": self.dst,
            "score": self.score,
            "temporal": self.temporal,
            "ctx": self.ctx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)


@dataclass
class Subgraph:
    """A directed heterogeneous entity-relation subgraph rooted at *seed*."""
    seed: str
    nodes: List[str] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Subgraph":
        return cls(
            seed=d["seed"],
            nodes=d["nodes"],
            edges=[Edge.from_dict(e) for e in d["edges"]],
        )

    # Convenience properties
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)




class GraphBuilder:
    """Constructs a validated entity-relation subgraph for a seed entity.

    Expansion alternates between KG neighbour lookup (probability alpha)
    and LLM-based entity discovery (probability 1-alpha).  Each candidate
    edge is scored by the LLM validator; edges below score_thresh are
    discarded.

    Implements Eq.(4)-(5) in the paper.

    Usage::

        builder = GraphBuilder(
            kg_client=my_kg,
            llm_discoverer=my_discoverer,
            edge_scorer=my_scorer,
        )
        subgraph = builder.build(seed="LHON", passages=[...])

    Args:
        kg_client: object with `.neighbors(entity) -> List[(neighbor, relation)]`.
            Wraps UMLS, Wikidata, or PubChem depending on domain.
        llm_discoverer: callable(entity, passages) -> List[(neighbor, relation)].
            Prompts the teacher model to identify domain-relevant neighbours
            beyond the static KG snapshot.
        edge_scorer: callable(src, relation, dst) -> float in [0, 1].
            Implements Eq.(5): combined relevance + domain-factual score.
        alpha: probability of using the KG branch (default 0.5, Eq.(4)).
        depth: number of iterative expansion rounds (default 3).
        score_thresh: minimum edge score to retain (default 0.5, Eq.(5)).
    """

    def __init__(
        self,
        kg_client: object,
        llm_discoverer: Callable,   # (entity, passages) -> [(nbr, rel)]
        edge_scorer: Callable,       # (src, rel, dst) -> float
        alpha: float = 0.5,
        depth: int = 3,
        score_thresh: float = 0.5,
    ):
        self.kg = kg_client
        self.discover = llm_discoverer
        self.score_edge = edge_scorer
        self.alpha = alpha
        self.depth = depth
        self.score_thresh = score_thresh

    def build(self, seed: str, passages: List[str]) -> Subgraph:
        """Build and return a validated subgraph rooted at *seed*.

        Args:
            seed: a rare seed entity string from 𝔼_rare.
            passages: top-5 TF-IDF-retrieved passages containing *seed*,
                passed to the LLM discoverer branch (Eq.(4)).

        Returns:
            Validated Subgraph with nodes and scored edges.
        """
        g = Subgraph(seed=seed, nodes=[seed])
        frontier = [seed]

        for _ in range(self.depth):
            next_frontier: List[str] = []
            for entity in frontier:
                for neighbor, relation in self._expand(entity, passages):
                    if neighbor in g.nodes:   # enforce simple-graph (no cycles)
                        continue
                    score = self.score_edge(entity, relation, neighbor)
                    if score >= self.score_thresh:
                        g.nodes.append(neighbor)
                        g.edges.append(Edge(
                            src=entity,
                            relation=relation,
                            dst=neighbor,
                            score=score,
                        ))
                        next_frontier.append(neighbor)
            frontier = next_frontier

        return g

    def _expand(
        self, entity: str, passages: List[str]
    ) -> List[Tuple[str, str]]:
        """Single expansion step: KG or LLM branch (Eq.(4))."""
        if random.random() < self.alpha:
            return self.kg.neighbors(entity)    # KG branch
        return self.discover(entity, passages)  # LLM branch




def make_llm_edge_scorer(
    teacher_fn: Callable,
    lam: float = 0.4,
) -> Callable[[str, str, str], float]:
    """Factory returning an edge scorer that implements Eq.(5).

    Score(e_i, r, e_j) = lambda * P_rel(e_i, r, e_j)
                       + (1-lambda) * P_dom(e_i, r, e_j)

    Args:
        teacher_fn: callable(prompt: str) -> dict with keys
            "p_rel" (float) and "p_dom" (float).
        lam: weighting factor lambda (default 0.4, from paper).

    Returns:
        Scorer callable compatible with GraphBuilder.
    """
    def _scorer(src: str, relation: str, dst: str) -> float:
        prompt = (
            f"Evaluate the triple ({src}, {relation}, {dst}).\n"
            "Return JSON with keys:\n"
            '  "p_rel": float[0,1]  # general relational plausibility\n'
            '  "p_dom": float[0,1]  # domain-specific factual correctness\n'
        )
        result = teacher_fn(prompt)
        p_rel = float(result.get("p_rel", 0.0))
        p_dom = float(result.get("p_dom", 0.0))
        return lam * p_rel + (1.0 - lam) * p_dom   # Eq.(5)

    return _scorer




class StubKGClient:
    """Minimal KG client for unit tests and demos (no external API needed).

    Stores a simple adjacency dict: {entity -> [(neighbor, relation), ...]}.
    """

    def __init__(self, graph: dict):
        """
        Args:
            graph: dict mapping entity -> list of (neighbor, relation) tuples.
        """
        self._g = graph

    def neighbors(self, entity: str) -> List[Tuple[str, str]]:
        return self._g.get(entity, [])
