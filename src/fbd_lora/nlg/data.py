"""NLG dataset loading and preprocessing for FBD-LoRA training.

Supports MetaMathQA and similar instruction-following datasets.
"""

import logging
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

ALPACA_MATH_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n{response}"
)

ALPACA_MATH_TEMPLATE_INFERENCE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n"
)


def format_alpaca_math(example: dict) -> str:
    """Apply the Alpaca math prompt template to a MetaMathQA example.

    Args:
        example: Dict with keys 'query' and 'response'.

    Returns:
        Formatted string for causal language modeling.
    """
    return ALPACA_MATH_TEMPLATE.format(
        query=example["query"],
        response=example["response"],
    )


def load_metamathqa(
    hf_path: str = "meta-math/MetaMathQA",
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Dataset:
    """Load MetaMathQA dataset from Hugging Face.

    Args:
        hf_path: HF dataset path.
        subset: Dataset subset/config name (None for default).
        split: Dataset split ('train', 'validation', etc.).
        max_samples: Maximum number of samples to load.
        hf_token: HuggingFace authentication token.

    Returns:
        HuggingFace Dataset object.
    """
    logger.info("Loading dataset: %s (split=%s, max_samples=%s)", hf_path, split, max_samples)

    load_kwargs = {"token": hf_token} if hf_token else {}
    if subset:
        dataset = load_dataset(hf_path, subset, split=split, **load_kwargs)
    else:
        dataset = load_dataset(hf_path, split=split, **load_kwargs)

    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))
        logger.info("Truncated dataset to %d samples.", n)

    logger.info("Loaded %d examples from %s.", len(dataset), hf_path)
    return dataset


def make_clm_collator(tokenizer):
    """Create a simple CLM data collator that pads to the max length in the batch.

    Pads input_ids and attention_mask with pad_token_id / 0.
    Sets labels to -100 at padding positions.

    Args:
        tokenizer: HuggingFace tokenizer with pad_token_id set.

    Returns:
        Callable collator function.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch):
        import torch
        max_len = max(len(ex["input_ids"]) for ex in batch)
        input_ids_out, attn_mask_out, labels_out = [], [], []

        for ex in batch:
            seq = ex["input_ids"]
            n = len(seq)
            pad_len = max_len - n

            input_ids_out.append(seq + [pad_id] * pad_len)
            attn_mask_out.append(ex.get("attention_mask", [1] * n) + [0] * pad_len)
            labels = ex.get("labels", seq)
            labels_out.append(list(labels) + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_out, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask_out, dtype=torch.long),
            "labels": torch.tensor(labels_out, dtype=torch.long),
        }

    return collate


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    prompt_template: str = "alpaca_math",
    num_proc: int = 4,
) -> Dataset:
    """Tokenize the dataset for causal language modeling.

    Applies template, tokenizes, creates labels (same as input_ids for CLM).
    Labels are -100 for the prompt portion (instruction masking).

    Args:
        dataset: Raw dataset with 'query' and 'response' fields.
        tokenizer: HuggingFace tokenizer with padding_side='right'.
        max_seq_length: Maximum sequence length (truncates longer sequences).
        prompt_template: Template name ('alpaca_math').
        num_proc: Number of processes for parallel tokenization.

    Returns:
        Tokenized dataset with 'input_ids', 'attention_mask', 'labels'.
    """
    if prompt_template != "alpaca_math":
        raise ValueError(f"Unsupported template: {prompt_template}")

    def tokenize_fn(example: dict) -> dict:
        full_text = format_alpaca_math(example)

        # Tokenize full text
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # For causal LM: labels = input_ids (standard CLM)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    logger.info("Tokenizing dataset with max_seq_length=%d...", max_seq_length)
    tokenized = dataset.map(
        tokenize_fn,
        batched=False,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    logger.info("Tokenization complete: %d examples.", len(tokenized))
    return tokenized
