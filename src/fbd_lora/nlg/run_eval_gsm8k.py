"""Standalone GSM8K evaluation script for FBD-LoRA trained models.

Usage:
    python -m fbd_lora.nlg.run_eval_gsm8k \
        --checkpoint <path_to_checkpoint_dir> \
        --base_model mistralai/Mistral-7B-v0.1 \
        --output_dir <path_to_output_eval_dir> \
        [--max_samples 1319] \
        [--batch_size 8] \
        [--max_new_tokens 512]
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ALPACA_MATH_TEMPLATE_INFERENCE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n"
)


def extract_answer_gsm8k(text: str) -> Optional[str]:
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return numbers[-1] if numbers else None


def exact_match_score(pred: str, ref: str) -> bool:
    pred = pred.replace(",", "").strip()
    ref = ref.replace(",", "").strip()
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except (ValueError, TypeError):
        return pred == ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to adapter checkpoint dir")
    parser.add_argument("--base_model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit eval samples (None = full GSM8K test)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    import os
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    logger.info("Loading tokenizer from %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, token=hf_token, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    logger.info("Loading base model %s (bf16)", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # Load adapter and merge
    logger.info("Loading adapter from %s", args.checkpoint)
    model = PeftModel.from_pretrained(model, args.checkpoint)
    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()
    model.eval()
    logger.info("Model ready. Params: %.1fB", sum(p.numel() for p in model.parameters()) / 1e9)

    # Load GSM8K test set
    logger.info("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    logger.info("Evaluating on %d examples", len(dataset))

    predictions_path = output_dir / "gsm8k_predictions.jsonl"
    metrics_path = output_dir / "gsm8k_metrics.json"

    correct = 0
    total = 0

    with open(predictions_path, "w") as f_out:
        for i in tqdm(range(len(dataset)), desc="GSM8K eval"):
            example = dataset[i]
            question = example["question"]
            reference = example["answer"]

            prompt = ALPACA_MATH_TEMPLATE_INFERENCE.format(query=question)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            pred_answer = extract_answer_gsm8k(generated)
            ref_answer = extract_answer_gsm8k(reference)

            is_correct = False
            if pred_answer is not None and ref_answer is not None:
                is_correct = exact_match_score(pred_answer, ref_answer)

            correct += int(is_correct)
            total += 1

            record = {
                "idx": i,
                "question": question,
                "reference": reference,
                "generated": generated,
                "pred_answer": pred_answer,
                "ref_answer": ref_answer,
                "correct": is_correct,
            }
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0:
                running_acc = 100.0 * correct / total
                logger.info("[%d/%d] Running accuracy: %.2f%%", i + 1, len(dataset), running_acc)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    metrics = {
        "gsm8k_exact_match": accuracy,
        "total": total,
        "correct": correct,
        "checkpoint": str(args.checkpoint),
        "base_model": args.base_model,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("=" * 50)
    logger.info("GSM8K Result: %.2f%% (%d / %d)", accuracy, correct, total)
    logger.info("Predictions: %s", predictions_path)
    logger.info("Metrics:     %s", metrics_path)
    logger.info("=" * 50)

    return metrics


if __name__ == "__main__":
    main()
