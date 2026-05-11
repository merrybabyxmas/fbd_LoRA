"""GSM8K and MATH evaluation for FBD-LoRA NLG models."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def extract_answer_gsm8k(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K model output.

    Looks for patterns like '#### 42' or 'The answer is 42'.
    """
    # GSM8K format: #### {answer}
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in text
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    return numbers[-1] if numbers else None


def exact_match_score(prediction: str, reference: str) -> bool:
    """Check if prediction matches reference (normalized)."""
    pred = prediction.replace(",", "").strip()
    ref = reference.replace(",", "").strip()
    try:
        return abs(float(pred) - float(ref)) < 1e-6
    except (ValueError, TypeError):
        return pred == ref


def evaluate_gsm8k(
    model,
    tokenizer,
    dataset,
    output_dir: str,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Evaluate model on GSM8K dataset.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer.
        dataset: Dataset with 'question' and 'answer' fields.
        output_dir: Directory to save raw generations and metrics.
        max_new_tokens: Maximum tokens to generate.
        batch_size: Evaluation batch size.
        device: Device string.

    Returns:
        Metrics dict with 'gsm8k_exact_match' in [0, 100].
    """
    from fbd_lora.nlg.data import ALPACA_MATH_TEMPLATE_INFERENCE

    model.eval()
    predictions_path = Path(output_dir) / "gsm8k_predictions.jsonl"
    metrics_path = Path(output_dir) / "gsm8k_metrics.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    results = []

    with open(predictions_path, "w") as f_out:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            questions = batch["question"] if "question" in batch else batch["query"]
            references = batch["answer"] if "answer" in batch else batch["response"]

            for q, ref in zip(questions, references):
                prompt = ALPACA_MATH_TEMPLATE_INFERENCE.format(query=q)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0 if not model.config.do_sample else 1.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated = tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                pred_answer = extract_answer_gsm8k(generated)
                ref_answer = extract_answer_gsm8k(ref)

                is_correct = False
                if pred_answer is not None and ref_answer is not None:
                    is_correct = exact_match_score(pred_answer, ref_answer)

                correct += int(is_correct)
                total += 1

                record = {
                    "question": q,
                    "reference": ref,
                    "generated": generated,
                    "pred_answer": pred_answer,
                    "ref_answer": ref_answer,
                    "correct": is_correct,
                }
                results.append(record)
                f_out.write(json.dumps(record) + "\n")

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    metrics = {"gsm8k_exact_match": accuracy, "total": total, "correct": correct}
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("GSM8K evaluation complete: %.2f%% (%d/%d)", accuracy, correct, total)
    return metrics
