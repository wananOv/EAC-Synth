"""
eac_synth/entity/idf_filter.py
IDF-based rare entity mining -- Stage 1 of EAC-Synth.

Implements Equations (2) and (3) from the paper:
    IDF(e, Omega) = log( N / (1 + df(e, Omega)) )           ... Eq.(2)
    tau_rare      = log( N / kappa ),  kappa = 10            ... Eq.(3)
"""

import math
from typing import Callable, Dict, List, Set, Tuple


def compute_corpus_idf(
    entity_to_docs: Dict[str, Set[int]],
    N: int,
) -> Dict[str, float]:
    """Compute smoothed IDF for every entity in the corpus.

    IDF(e, Omega) = log( N / (1 + df(e)) )  -- Eq.(2) in paper.

    Args:
        entity_to_docs: maps canonical entity string to the set of document
            IDs in which it appears (at least one mention).
        N: total number of documents in the corpus.

    Returns:
        Dict mapping entity -> IDF score.
    """
    return {
        e: math.log(N / (1.0 + len(doc_ids)))
        for e, doc_ids in entity_to_docs.items()
    }


def select_rare_entities(
    idf: Dict[str, float],
    N: int,
    ontology_check: Callable[[str], bool],
    kappa: int = 10,
) -> List[Tuple[str, float]]:
    """Return rare seed entities above tau_rare, sorted by IDF descending.

    Implements Eq.(3): tau_rare = log(N / kappa).

    Applies ontology validation to discard typographic errors and
    non-standard abbreviations that pass the IDF filter.

    Args:
        idf: entity -> IDF score mapping (output of compute_corpus_idf).
        N: corpus document count.
        ontology_check: callable(entity) -> bool; True iff the entity maps
            to a valid concept in the domain ontology.
        kappa: document-frequency ceiling (default 10, i.e. entities that
            appear in fewer than kappa documents are considered rare).

    Returns:
        Sorted list of (entity, idf_score) tuples (descending by score).
    """
    tau_rare = math.log(N / kappa)  # Eq.(3) threshold
    candidates = [
        (e, s)
        for e, s in idf.items()
        if s >= tau_rare and ontology_check(e)  # IDF gate + ontology gate
    ]
    return sorted(candidates, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Convenience: build entity_to_docs from a tokenised corpus
# ---------------------------------------------------------------------------

def build_entity_doc_index(
    documents: List[List[str]],
    entity_set: Set[str],
) -> Dict[str, Set[int]]:
    """Build the entity -> {doc_id, ...} index from a pre-tokenised corpus.

    Args:
        documents: list of token lists, one per document.
        entity_set: set of canonical entity strings to track.

    Returns:
        Dict[entity, Set[doc_id]].
    """
    index: Dict[str, Set[int]] = {e: set() for e in entity_set}
    for doc_id, tokens in enumerate(documents):
        token_set = set(tokens)
        for entity in entity_set:
            if entity in token_set:
                index[entity].add(doc_id)
    return index
