"""
eac_synth/graph/chain_extractor.py
Longest-Chain Extraction -- Stage 3 of EAC-Synth.

Implements Equation (7) from the paper:
    P* = argmax_{P in Pi(G_s)} |P|   s.t.  SemValid(P, M_T) = 1

Algorithm:
    1. Enumerate all simple directed paths by iterative DFS.
    2. Sort candidates by decreasing length.
    3. Score the top-K with the LLM validity scorer.
    4. Return the first (longest) path with validity score >= threshold.

Paths shorter than min_hops (default 2) are discarded as insufficiently
complex.  Across all three domains the paper reports 89.4% of subgraphs
yield a valid P* with an average length of 4.7 hops.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from eac_synth.graph.graph_builder import Edge, Subgraph




_SYSTEM_PROMPT = (
    "You are a domain expert. A reasoning chain is semantically valid when:\n"
    "(a) each stated relation holds between its endpoints within the\n"
    "    given domain and temporal context, and\n"
    "(b) the complete chain leads to a single unambiguous terminal conclusion.\n"
)

_USER_TEMPLATE = (
    "Domain: {domain}\n\n"
    "Chain: {chain}\n\n"
    'Return JSON: {{"valid": bool, "score": float[0,1], "reason": str}}'
)


def format_chain_for_prompt(path: List[Edge]) -> str:
    """Render a path as a human-readable chain string for the LLM prompt.

    Example output::
        LHON -[caused_by, ~1988]-> MT-ND4 -[encodes]-> NADH dehydrogenase
    """
    parts = []
    for edge in path:
        tag = f", {edge.temporal}" if edge.temporal else ""
        parts.append(f"{edge.src} -[{edge.relation}{tag}]-> {edge.dst}")
    # Merge into a single chain string by sharing endpoints
    if not parts:
        return ""
    tokens = [path[0].src]
    for edge in path:
        label = f"[{edge.relation}" + (f", {edge.temporal}" if edge.temporal else "") + "]"
        tokens += [f"-{label}->", edge.dst]
    return " ".join(tokens)


class LongestChainExtractor:
    """Extracts the longest semantically valid simple directed path.

    Implements Eq.(7) from the paper::

        P* = argmax_{P in Pi(G_s)} |P|   s.t.  SemValid(P, M_T) = 1

    Args:
        llm_scorer: callable(path: List[Edge], domain: str) -> dict.
            Must return a dict with at least keys "valid" (bool) and
            "score" (float in [0,1]).  Corresponds to SemValid in Eq.(7).
        domain: domain label passed to the LLM (e.g. "medical", "legal").
        top_k: maximum number of candidate paths scored by the LLM per
            subgraph (default 5, to limit API calls).
        min_hops: minimum path length in edges; shorter paths are discarded
            (default 2 -- one-hop questions are too easy, §3.4).
        score_thresh: minimum LLM validity score to accept a path (0.7).
    """

    def __init__(
        self,
        llm_scorer: Callable[[List[Edge], str], Dict[str, Any]],
        domain: str,
        top_k: int = 5,
        min_hops: int = 2,
        score_thresh: float = 0.7,
    ):
        self.score = llm_scorer
        self.domain = domain
        self.top_k = top_k
        self.min_hops = min_hops
        self.score_thresh = score_thresh

    def extract(self, subgraph: Subgraph) -> Optional[List[Edge]]:
        """Return the best path as List[Edge], or None if none qualifies.

        Args:
            subgraph: validated Subgraph from Stage 2.

        Returns:
            Longest semantically valid path, or None if the subgraph is
            skipped (no path meets the validity threshold).
        """
        all_paths = self._dfs_enumerate(subgraph)
        eligible = [p for p in all_paths if len(p) >= self.min_hops]

        if not eligible:
            return None

        # Score only the K longest paths to minimise LLM API calls
        for path in sorted(eligible, key=len, reverse=True)[: self.top_k]:
            result = self.score(path, self.domain)
            if result.get("valid") and result.get("score", 0) >= self.score_thresh:
                return path

        return None  # subgraph skipped

    @staticmethod
    def _dfs_enumerate(subgraph: Subgraph) -> List[List[Edge]]:
        """Enumerate all simple directed paths via DFS.

        A path is *simple* iff it visits no node more than once.
        Feasible because |V_s| <= 25 in practice (paper §3.4).

        Returns:
            List of paths, each path being a List[Edge].
        """
        adj: dict = {}
        for edge in subgraph.edges:
            adj.setdefault(edge.src, []).append(edge)

        collected: List[List[Edge]] = []

        def dfs(node: str, path: List[Edge], visited: set) -> None:
            if path:
                collected.append(list(path))
            for edge in adj.get(node, []):
                if edge.dst not in visited:
                    visited.add(edge.dst)
                    path.append(edge)
                    dfs(edge.dst, path, visited)
                    path.pop()
                    visited.discard(edge.dst)

        for start in subgraph.nodes:
            dfs(start, [], {start})

        return collected




def make_llm_chain_scorer(
    teacher_fn: Callable[[str, str], Dict[str, Any]],
) -> Callable[[List[Edge], str], Dict[str, Any]]:
    """Factory returning an LLM scorer compatible with LongestChainExtractor.

    Args:
        teacher_fn: callable(system_prompt, user_prompt) -> dict.
            The teacher model should return parsed JSON with keys
            "valid", "score", and "reason".

    Returns:
        Scorer callable: (path, domain) -> dict.
    """
    def _scorer(path: List[Edge], domain: str) -> Dict[str, Any]:
        chain_str = format_chain_for_prompt(path)
        user_msg = _USER_TEMPLATE.format(domain=domain, chain=chain_str)
        return teacher_fn(_SYSTEM_PROMPT, user_msg)

    return _scorer
