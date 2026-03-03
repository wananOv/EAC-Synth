"""
eac_synth/training/train_lora.py
LoRA fine-tuning entry point -- Stage 5 of EAC-Synth.

Fine-tunes LLaMA-3-8B-Instruct with LoRA adapters on the synthesized
EAC-Synth dataset.  Implements the training objective in Eq.(1) of the paper:

    L(theta) = -1/|Omega_synth| * sum_i sum_t log p_theta([c_i; y_i]_t | x_i, [c_i; y_i]_{<t})

where the chain-of-thought reasoning chain c_i is concatenated with the
answer y_i as a structured prefix in the assistant turn.

LoRA configuration (paper §4.3):
    rank r      = 16
    alpha       = 32
    dropout     = 0.05
    target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    trainable params: ~41.9M / 8B total (0.52%)

Optimisation (paper §4.3):
    AdamW: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01
    Cosine annealing LR: peak=2e-4, min=1e-6, warmup=120 steps
    Epochs: 5
    Effective batch: 32 (micro 4 x grad_accum 8)
    Mixed precision: bfloat16
    Gradient clipping: 1.0

Hardware used in paper: 4x NVIDIA A100-80GB with NVLink (~9.2h total).

Usage::

    python -m eac_synth.training.train_lora \\
        --data  data/eac_synth_medical.json \\
        --model meta-llama/Meta-Llama-3-8B-Instruct \\
        --output checkpoints/medical_lora
"""

import json
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

LORA = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
)


# ---------------------------------------------------------------------------
# Sample formatting
# ---------------------------------------------------------------------------

def format_sample(s: dict, tok) -> dict:
    """Tokenise one EAC-Synth sample into a training instance.

    Chain-of-thought reasoning steps are prepended to the answer
    in the assistant turn, implementing Eq.(1): CoT-conditioned loss.

    Args:
        s: sample dict with keys "question", "steps" (list[str]), "answer".
        tok: HuggingFace tokenizer.

    Returns:
        Dict with "input_ids" and "labels" tensors (labels == input_ids
        for causal LM training -- the collator handles masking).
    """
    cot = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(s["steps"]))
    text = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{s['question']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{cot}\n\nFinal Answer: {s['answer']}<|eot_id|>"
    )
    enc = tok(text, truncation=True, max_length=1024, padding=False)
    enc["labels"] = enc["input_ids"].copy()
    return enc


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main(data_path: str, base_model: str, output_dir: str) -> None:
    """Load data, attach LoRA, and run training.

    Args:
        data_path: path to EAC-Synth JSON file (list of sample dicts).
        base_model: HuggingFace model identifier or local path.
        output_dir: directory to save the final LoRA adapter weights.
    """
    # Tokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token

    # Base model (bfloat16 for A100 efficiency)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Attach LoRA adapters
    model = get_peft_model(model, LORA)
    model.print_trainable_parameters()  # expect ~41.9M / 8B

    # Dataset
    data = json.load(open(data_path))
    dataset = Dataset.from_list([format_sample(s, tok) for s in data])

    # Training arguments (paper §4.3)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,   # effective batch = 32
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=120,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="tensorboard",
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    print(f"Saved LoRA adapter and tokenizer to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="EAC-Synth LoRA fine-tuning")
    ap.add_argument("--data",  required=True,
                    help="Path to EAC-Synth JSON dataset")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                    help="Base model identifier or local path")
    ap.add_argument("--output", required=True,
                    help="Output directory for LoRA adapter weights")
    a = ap.parse_args()

    main(a.data, a.model, a.output)
