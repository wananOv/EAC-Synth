"""
examples/demo_medical.py
Minimal end-to-end EAC-Synth demo using synthetic (offline) data.

This demo requires NO external API keys or GPU.  It uses:
  - A toy in-memory corpus (10 fake medical documents)
  - StubKGClient for KG lookups
  - A mock teacher_fn that returns plausible-looking JSON

Run::

    cd EAC-Synth
    python examples/demo_medical.py

Expected output: a small JSON corpus saved to demo_output/demo_corpus.json
"""

import json
import random
import sys
from pathlib import Path
# Make sure the package root is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from eac_synth.entity.idf_filter import compute_corpus_idf, select_rare_entities
from eac_synth.graph.graph_builder import GraphBuilder, Subgraph, StubKGClient
from eac_synth.graph.chain_extractor import LongestChainExtractor
from eac_synth.synthesis.verbalizer import Verbalizer
from eac_synth.synthesis.rcdc import RCDCFilter
from eac_synth.pipeline import EACSynthPipeline




DOCUMENTS = [
    ["LHON", "mitochondria", "MT-ND4", "vision", "loss", "optic"],
    ["Barth", "syndrome", "cardiolipin", "tafazzin", "cardiomyopathy"],
    ["MERRF", "myoclonic", "epilepsy", "MT-TK", "mitochondria"],
    ["NARP", "neuropathy", "ataxia", "MT-ATP6", "retinitis"],
    ["MELAS", "stroke-like", "episodes", "MT-TL1", "lactic", "acidosis"],
    ["LHON", "optic", "neuropathy", "gene", "MT-ND1"],
    ["Kearns-Sayre", "syndrome", "ptosis", "heart", "block"],
    ["CPEO", "chronic", "progressive", "external", "ophthalmoplegia"],
    ["Alpers", "syndrome", "POLG", "hepatopathy", "neurodegeneration"],
    ["Friedreich", "ataxia", "frataxin", "iron", "cardiomyopathy"],
]

# All unique tokens across documents
ALL_ENTITIES = {tok for doc in DOCUMENTS for tok in doc}

# Build entity -> doc_id index
entity_to_docs = {
    e: {i for i, doc in enumerate(DOCUMENTS) if e in doc}
    for e in ALL_ENTITIES
}

N = len(DOCUMENTS)




KG = {
    "LHON": [("MT-ND4", "encoded_by"), ("MT-ND1", "encoded_by"), ("optic", "affects")],
    "MT-ND4": [("mitochondria", "located_in"), ("vision", "impairs")],
    "optic": [("vision", "mediates"), ("loss", "causes")],
    "MERRF": [("MT-TK", "encoded_by"), ("myoclonic", "causes")],
    "MT-TK": [("mitochondria", "located_in")],
}

kg = StubKGClient(KG)




def mock_teacher(system: str, user: str, temperature: float = 0.0, top_p: float = 1.0) -> dict:
    """Returns plausible-looking mock responses for all teacher calls."""

    # Edge scoring
    if '"p_rel"' in user or "p_rel" in user:
        return {"p_rel": round(random.uniform(0.6, 0.9), 2),
                "p_dom": round(random.uniform(0.6, 0.9), 2)}

    # Chain validity scoring
    if '"valid"' in user or "valid" in user:
        return {"valid": True, "score": round(random.uniform(0.7, 0.95), 2),
                "reason": "All relations are domain-valid."}

    # LLM neighbour discovery
    if "neighbors" in user or "related entities" in user:
        return {"neighbors": [
            {"entity": "mitochondrial_dysfunction", "relation": "causes"},
            {"entity": "reactive_oxygen_species", "relation": "produces"},
        ]}

    # Verbalization
    if '"question"' in user or "question" in user:
        return {
            "question": (
                "A patient presents with bilateral central scotomas and impaired "
                "colour vision. His maternal relatives show similar symptoms. "
                "The culprit gene encodes a subunit of Complex I. "
                "Which mitochondrial gene is most likely mutated, and via which "
                "pathway does its dysfunction lead to optic nerve degeneration?"
            ),
            "steps": [
                "Step 1: The maternal inheritance pattern suggests a mitochondrial disorder.",
                "Step 2: Complex I subunit dysfunction in retinal ganglion cells impairs ATP synthesis.",
                "Step 3: Energy deficit triggers apoptosis of optic nerve fibres.",
                "Step 4: The MT-ND4 gene encodes the ND4 subunit of Complex I.",
            ],
            "answer": "MT-ND4",
        }

    return {}




def main():
    print("=== EAC-Synth Demo (offline / no API) ===\n")

    pipeline = EACSynthPipeline.from_teacher(
        teacher_fn=mock_teacher,
        domain="medical",
        kg_client=kg,
        kappa=5,          # lower kappa for tiny corpus
        depth=2,
        chain_min_hops=1,  # allow 1-hop for demo
        chain_score_thresh=0.5,
        rcdc_theta=0.80,
    )

    corpus = pipeline.run(
        corpus_docs=DOCUMENTS,
        entity_to_docs=entity_to_docs,
        N=N,
        verbose=True,
    )

    output_path = "demo_output/demo_corpus.json"
    EACSynthPipeline.save(corpus, output_path)

    print("\nSample entry:")
    if corpus:
        sample = corpus[0]
        print(f"  Question: {sample['question'][:120]}...")
        print(f"  Steps:    {len(sample['steps'])} reasoning steps")
        print(f"  Answer:   {sample['answer']}")
    print(f"\nTotal synthesised samples: {len(corpus)}")


if __name__ == "__main__":
    main()
