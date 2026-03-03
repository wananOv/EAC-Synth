# EAC-Synth: Entity-Anchored Corpus Synthesis for Low-Resource Domain LLM Fine-Tuning

> **Paper:** *EAC-Synth: Entity-Anchored Corpus Synthesis for Low-Resource Domain LLM Fine-Tuning via Knowledge-Graph Longest-Chain Extraction*  
> Ruohan Liu · Mingze Qian · Sichen Tang · Yufei Zhao · Haotian Chen  
> Southeast University / Tsinghua University / Fudan University, 2025

---

## Overview

EAC-Synth is a **four-stage, annotation-free** pipeline that constructs domain-specific fine-tuning corpora for low-resource LLM specialisation:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `entity/idf_filter.py` | IDF-based rare entity mining + ontology validation (Eq. 2–3) |
| 2 | `graph/graph_builder.py` | Directed entity-relation subgraph construction via hybrid KG+LLM expansion (Eq. 4–6) |
| 3 | `graph/chain_extractor.py` | Longest semantically valid path via DFS + LLM scoring (Eq. 7) |
| 4 | `synthesis/verbalizer.py` + `synthesis/rcdc.py` | Path→QA verbalization gated by Relation-Conditioned Diversity Constraint (Eq. 8–9) |

Fine-tuning LLaMA-3-8B-Instruct with 11,300 synthesised samples achieves:

- **MedQA-USMLE**: +5.3 pp over zero-shot  
- **LegalBench-Hard**: +4.8 pp  
- **SciQA-Expert**: +6.1 pp  

Entity rarity (IDF) correlates strongly with training-signal density (Pearson *r* = 0.74, *p* < 0.001).

---

## Repository Layout

```
EAC-Synth/
├── eac_synth/
│   ├── entity/
│   │   ├── idf_filter.py        # Stage 1: IDF computation + rare-entity selection
│   │   └── ner_extractor.py     # Stage 1: Domain NER + ontology normalisation
│   ├── graph/
│   │   ├── graph_builder.py     # Stage 2: Subgraph construction (Eq. 4-6)
│   │   └── chain_extractor.py   # Stage 3: Longest-chain DFS + LLM scoring (Eq. 7)
│   ├── synthesis/
│   │   ├── verbalizer.py        # Stage 4a: Path → (question, steps, answer)
│   │   └── rcdc.py              # Stage 4b: Diversity constraint (Eq. 8-9)
│   ├── training/
│   │   └── train_lora.py        # LoRA fine-tuning (LLaMA-3-8B-Instruct)
│   ├── evaluate/
│   │   └── run_eval.py          # Benchmark evaluation harness
│   └── pipeline.py              # End-to-end orchestrator
├── examples/
│   └── demo_medical.py          # Offline demo (no API / GPU required)
├── tests/
│   └── test_core.py             # Pytest unit tests
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/seu-nlp/EAC-Synth
cd EAC-Synth
pip install -r requirements.txt
```

### 2. Run the offline demo (no API key needed)

```bash
python examples/demo_medical.py
```

### 3. Run unit tests

```bash
pytest tests/test_core.py -v
```

---

## Full Pipeline Usage

```python
from eac_synth.pipeline import EACSynthPipeline
from eac_synth.synthesis.verbalizer import make_openai_teacher
import os

# 1. Configure teacher model
teacher = make_openai_teacher(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 2. Build pipeline (medical domain)
pipeline = EACSynthPipeline.from_teacher(teacher, domain="medical")

# 3. Run synthesis
#    corpus_docs      : List[List[str]]  -- tokenised documents
#    entity_to_docs   : Dict[str, Set[int]] -- entity -> doc IDs
#    N                : int -- total document count
corpus = pipeline.run(
    corpus_docs=my_docs,
    entity_to_docs=entity_to_docs,
    N=4200,
    verbose=True,
)

# 4. Save
EACSynthPipeline.save(corpus, "data/eac_synth_medical.json")
```

---

## Fine-Tuning

```bash
python -m eac_synth.training.train_lora \
    --data  data/eac_synth_medical.json \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --output checkpoints/medical_lora
```

**LoRA config** (paper §4.3): rank=16, alpha=32, dropout=0.05, target modules: q/k/v/o/gate/up/down_proj.  
**Optimizer**: AdamW with cosine LR schedule, peak lr=2e-4, 5 epochs, effective batch=32, bfloat16.

---

## Evaluation

```bash
python -m eac_synth.evaluate.run_eval \
    --model  checkpoints/medical_lora \
    --benchmark medqa \
    --data   data/MedQA-USMLE-test.jsonl \
    --output results/medqa_eval.json
```

Supported benchmarks: `medqa`, `legalbench`, `sciq`.

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kappa` | 10 | Document-frequency ceiling for rare entities (Eq. 3) |
| `alpha` | 0.5 | KG vs LLM expansion probability (Eq. 4) |
| `depth` | 3 | Subgraph expansion depth |
| `edge_score_thresh` | 0.5 | Minimum edge score to retain (Eq. 5) |
| `chain_min_hops` | 2 | Minimum path length (hops) |
| `chain_score_thresh` | 0.7 | Minimum LLM validity score for path (Eq. 7) |
| `rcdc_gamma` | 0.5 | Structural vs lexical weight in RCDC (Eq. 8) |
| `rcdc_theta` | 0.75 | RCDC acceptance threshold (Eq. 9) |

---

## Citation

```bibtex
@article{liu2025eacsynth,
  title   = {EAC-Synth: Entity-Anchored Corpus Synthesis for Low-Resource Domain LLM Fine-Tuning via Knowledge-Graph Longest-Chain Extraction},
  author  = {Liu, Ruohan and Qian, Mingze and Tang, Sichen and Zhao, Yufei and Chen, Haotian},
  year    = {2025},
}
```
