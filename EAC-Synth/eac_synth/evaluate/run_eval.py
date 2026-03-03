"""
eac_synth/evaluate/run_eval.py
Benchmark evaluation harness.

Evaluates a fine-tuned model (or base model) on the three held-out domain
benchmarks used in the paper:
    - MedQA-USMLE  (1273 questions, 4-option multiple choice)
    - LegalBench-Hard (600 questions)
    - SciQA-Expert    (920 questions)

Metrics: 0-shot accuracy.  All reported improvements satisfy p < 0.01
(McNemar's test, Holm-Bonferroni corrected) -- paper §4.3.

Usage::

    python -m eac_synth.evaluate.run_eval \\
        --model  checkpoints/medical_lora \\
        --benchmark medqa \\
        --output results/medqa_eval.json

    python -m eac_synth.evaluate.run_eval \\
        --model  checkpoints/legal_lora \\
        --benchmark legalbench \\
        --output results/legal_eval.json
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from scipy.stats import mcnemar
from statsmodels.stats.multitest import multipletests




def load_model_and_tokenizer(model_path: str):
    """Load a fine-tuned LoRA model or a plain base model.

    If *model_path* contains a LoRA adapter (adapter_config.json present),
    the PEFT merge path is used; otherwise the model is loaded as-is.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token

    adapter_cfg = Path(model_path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        # Derive base model name from adapter config
        cfg = json.loads(adapter_cfg.read_text())
        base_name = cfg.get("base_model_name_or_path",
                            "meta-llama/Meta-Llama-3-8B-Instruct")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    model.eval()
    return model, tok


@torch.no_grad()
def generate_answer(
    model,
    tok,
    question: str,
    max_new_tokens: int = 256,
) -> str:
    """Generate a 0-shot answer string for *question*."""
    prompt = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()




_LETTER_RE = re.compile(r"\b([A-Da-d])\b")


def extract_choice_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from generated text."""
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


def check_correct(prediction: str, gold: str, task_type: str) -> bool:
    """Return True if prediction matches the gold answer.

    Args:
        prediction: raw model output string.
        gold: gold answer string.
        task_type: "mcq" (multiple choice A-D) or "free" (substring match).
    """
    if task_type == "mcq":
        pred_letter = extract_choice_letter(prediction)
        gold_letter = extract_choice_letter(gold)
        return pred_letter == gold_letter
    else:  # free-form substring match
        return gold.strip().lower() in prediction.lower()




def load_medqa(path: str) -> List[Dict[str, Any]]:
    """Load MedQA-USMLE jsonl.  Expected keys: question, options, answer."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # Build MCQ string
            opts = "  ".join(
                f"({k}) {v}" for k, v in item["options"].items()
            )
            data.append({
                "question": f"{item['question']}\n{opts}",
                "gold": item["answer"],
                "task_type": "mcq",
            })
    return data


def load_legalbench(path: str) -> List[Dict[str, Any]]:
    """Load LegalBench-Hard jsonl.  Expected keys: input, output."""
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "question": item["input"],
                "gold": item["output"],
                "task_type": "mcq",
            })
    return data


def load_sciq(path: str) -> List[Dict[str, Any]]:
    """Load SciQA-Expert json.  Expected keys: question, correct_answer."""
    items = json.load(open(path))
    return [
        {
            "question": it["question"],
            "gold": it["correct_answer"],
            "task_type": "free",
        }
        for it in items
    ]


BENCHMARK_LOADERS = {
    "medqa": load_medqa,
    "legalbench": load_legalbench,
    "sciq": load_sciq,
}




def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> Tuple[float, float]:
    """McNemar's test comparing two systems on the same test set.

    Args:
        correct_a, correct_b: boolean correctness vectors of equal length.

    Returns:
        (statistic, p_value)
    """
    assert len(correct_a) == len(correct_b)
    n01 = sum(not a and b for a, b in zip(correct_a, correct_b))
    n10 = sum(a and not b for a, b in zip(correct_a, correct_b))
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=True)
    return float(result.statistic), float(result.pvalue)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return boolean mask of significant tests after Holm-Bonferroni correction."""
    _, corrected, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return list(corrected)




def evaluate(
    model,
    tok,
    dataset: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run 0-shot evaluation on *dataset*.

    Returns a result dict with accuracy, per-sample predictions, etc.
    """
    predictions, correct_flags = [], []

    for i, item in enumerate(dataset):
        pred = generate_answer(model, tok, item["question"])
        correct = check_correct(pred, item["gold"], item["task_type"])
        predictions.append(pred)
        correct_flags.append(correct)

        if verbose and (i + 1) % 50 == 0:
            acc = sum(correct_flags) / len(correct_flags) * 100
            print(f"  [{i+1}/{len(dataset)}]  running accuracy: {acc:.1f}%")

    accuracy = sum(correct_flags) / len(correct_flags) * 100
    return {
        "accuracy": accuracy,
        "n_correct": sum(correct_flags),
        "n_total": len(dataset),
        "predictions": predictions,
        "correct_flags": correct_flags,
    }




def main() -> None:
    ap = argparse.ArgumentParser(description="EAC-Synth benchmark evaluation")
    ap.add_argument("--model",     required=True, help="Model path or HF identifier")
    ap.add_argument("--benchmark", required=True, choices=list(BENCHMARK_LOADERS),
                    help="Benchmark name")
    ap.add_argument("--data",      required=True, help="Path to benchmark data file")
    ap.add_argument("--output",    required=True, help="Path to write result JSON")
    ap.add_argument("--verbose",   action="store_true", default=True)
    args = ap.parse_args()

    print(f"Loading model from {args.model} ...")
    model, tok = load_model_and_tokenizer(args.model)

    print(f"Loading benchmark: {args.benchmark} from {args.data} ...")
    dataset = BENCHMARK_LOADERS[args.benchmark](args.data)
    print(f"  {len(dataset)} questions loaded.")

    print("Running evaluation ...")
    result = evaluate(model, tok, dataset, verbose=args.verbose)

    print(f"\nAccuracy: {result['accuracy']:.1f}%  "
          f"({result['n_correct']}/{result['n_total']})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
