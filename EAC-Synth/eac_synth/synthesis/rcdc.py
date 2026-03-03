"""
eac_synth/synthesis/rcdc.py
Relation-Conditioned Diversity Constraint -- Stage 4 filter.

Implements Equations (8) and (9) from the paper:

    Phi(s_new, s) = gamma * Sim_rho(rho(s_new), rho(s))
                  + (1-gamma) * Sim_3g(x_new, x_s)       ... Eq.(8)

    Accept s_new  iff  max_{s in pool} Phi(s_new, s) < theta_div   ... Eq.(9)

where:
    Sim_rho  = normalized SequenceMatcher edit-distance ratio on relation tuples
    Sim_3g   = Jaccard similarity of character 3-gram sets on question strings
    gamma    = 0.5  (equal weighting)
    theta_div = 0.75 (selected by grid-search, paper §3.5.2)

RCDC rejects an average of 23.1% of raw candidates (18% scientific,
31% legal) -- paper Table 1.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple




def jaccard_3g(a: str, b: str) -> float:
    """Character-3-gram Jaccard similarity between two strings.

    Falls back to exact-match if either string is shorter than 3 chars.

    Args:
        a, b: question strings to compare.

    Returns:
        Jaccard similarity in [0, 1].
    """
    def g3(s: str) -> set:
        return set(s[i: i + 3] for i in range(len(s) - 2)) or {s}

    sa, sb = g3(a.lower()), g3(b.lower())
    return len(sa & sb) / len(sa | sb)


def rel_sim(r1: Tuple[str, ...], r2: Tuple[str, ...]) -> float:
    """Normalised edit-distance similarity between relation label sequences.

    Uses Python's difflib.SequenceMatcher ratio, which equals
    2 * M / T  where M = number of matching characters and T = total.

    Args:
        r1, r2: ordered tuples of relation label strings.

    Returns:
        Similarity ratio in [0, 1].
    """
    return SequenceMatcher(None, r1, r2).ratio()


def rel_seq(sample: dict) -> Tuple[str, ...]:
    """Extract the ordered relation-label sequence from a sample's path.

    Args:
        sample: dict with key "path" containing a list of edge dicts,
            each with key "relation".

    Returns:
        Tuple of relation label strings, e.g. ("causes", "inhibits", ...).
    """
    return tuple(edge["relation"] for edge in sample["path"])




class RCDCFilter:
    """Relation-Conditioned Diversity Constraint.

    Accepts s_new only if::

        max_{s in pool} Phi(s_new, s) < theta_div

    where Phi = gamma * Sim_rel + (1-gamma) * Sim_3g
    implements Eq.(8)-(9) of the paper.

    Accepted samples are automatically added to the internal pool so
    subsequent calls reflect the growing corpus.

    Args:
        gamma: weighting between structural and lexical similarity
            (default 0.5, paper §3.5.2).
        theta_div: acceptance threshold; lower = stricter diversity
            (default 0.75, paper §3.5.2).

    Example::

        filt = RCDCFilter()
        for sample in raw_candidates:
            if filt.accept(sample):
                final_corpus.append(sample)
        print(f"Kept {filt.pool_size} / {len(raw_candidates)} samples")
    """

    def __init__(self, gamma: float = 0.5, theta_div: float = 0.75):
        self.gamma = gamma
        self.theta_div = theta_div
        self._pool: List[dict] = []

    def accept(self, sample: dict) -> bool:
        """Return True iff *sample* passes the diversity constraint.

        Accepted samples are appended to the internal pool.

        Args:
            sample: dict with at least keys:
                "question" (str)   -- verbalized question text
                "path"     (list)  -- list of edge dicts with "relation" key

        Returns:
            True if accepted; False if rejected (too similar to existing sample).
        """
        r_new = rel_seq(sample)
        q_new: str = sample["question"]

        for stored in self._pool:
            phi = (
                self.gamma * rel_sim(r_new, rel_seq(stored))
                + (1.0 - self.gamma) * jaccard_3g(q_new, stored["question"])
            )
            if phi >= self.theta_div:
                return False   # too similar -> reject  (Eq.9)

        self._pool.append(sample)
        return True

    def batch_filter(self, candidates: List[dict]) -> List[dict]:
        """Convenience method: filter an iterable of candidates.

        Args:
            candidates: list of sample dicts (see accept()).

        Returns:
            Subset of candidates that passed the diversity constraint,
            in the original order.
        """
        return [s for s in candidates if self.accept(s)]

    @property
    def pool_size(self) -> int:
        """Number of accepted samples in the internal pool."""
        return len(self._pool)

    def reset(self) -> None:
        """Clear the internal pool (useful between domain runs)."""
        self._pool.clear()
