"""NLG dataset loading and preprocessing for FBD-LoRA training.

Supports:
  - fxmeng/pissa-dataset (PiSSA protocol: metamath, python, conversation sub-tasks)
  - meta-math/MetaMathQA (legacy fallback)
  - Other instruction-following datasets with instruction/input/output or query/response fields.

PiSSA dataset sub-tasks:
  - "metamath"     : Filter by type in {GSM_Rephrased, GSM_AnsAug, GSM_SV, GSM_FOBAR,
                                         MATH_AnsAug, MATH_Rephrased, MATH_FOBAR, MATH_SV}
  - "python"       : Filter by type == "python"
  - "conversation" : Filter by type == "conversation"
  - None / "all"   : Use full dataset

Column mapping (fxmeng/pissa-dataset): instruction + input -> query, output -> response
"""

import logging
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n{response}"
)

ALPACA_TEMPLATE_INFERENCE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query}\n\n### Response:\n"
)

# Legacy alias used by old configs
ALPACA_MATH_TEMPLATE = ALPACA_TEMPLATE
ALPACA_MATH_TEMPLATE_INFERENCE = ALPACA_TEMPLATE_INFERENCE


# Metamath type values from fxmeng/pissa-dataset
_METAMATH_TYPES = frozenset({
    "GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR",
    "MATH_AnsAug", "MATH_Rephrased", "MATH_FOBAR", "MATH_SV",
})

_PISSA_SUB_TASK_TYPES = {
    "metamath": _METAMATH_TYPES,
    "python": frozenset({"python"}),
    "conversation": frozenset({"conversation"}),
}

# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def format_alpaca(example: dict) -> str:
    """Apply the Alpaca prompt template.

    Handles both MetaMathQA schema (query/response) and
    pissa-dataset schema (instruction/input/output, pre-mapped to query/response).

    Args:
        example: Dict with keys 'query' and 'response'.

    Returns:
        Formatted string for causal language modeling.
    """
    return ALPACA_TEMPLATE.format(
        query=example["query"],
        response=example["response"],
    )


# Legacy alias
format_alpaca_math = format_alpaca


def _pissa_to_standard(example: dict) -> dict:
    """Map fxmeng/pissa-dataset fields to standard query/response schema.

    pissa-dataset columns: instruction, input, output, type
    Standard columns:     query, response

    If input is non-empty, append it to the instruction.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()

    if inp:
        query = f"{instruction}\n\n{inp}"
    else:
        query = instruction

    return {
        "query": query,
        "response": output,
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_pissa_dataset(
    sub_task: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Dataset:
    """Load fxmeng/pissa-dataset with optional sub_task filtering.

    The dataset has a single HF config "default" with a 'type' field
    that encodes the sub-task category.

    Args:
        sub_task: One of "metamath", "python", "conversation", or None/all.
                  Maps to the 'type' field values in the dataset.
        split: Dataset split ('train', 'test').
        max_samples: Maximum number of samples to use.
        hf_token: HuggingFace authentication token.

    Returns:
        HuggingFace Dataset with 'query' and 'response' columns.
    """
    hf_path = "fxmeng/pissa-dataset"
    logger.info(
        "Loading pissa-dataset from HF (sub_task=%s, split=%s, max_samples=%s)",
        sub_task, split, max_samples,
    )

    load_kwargs = {"token": hf_token} if hf_token else {}
    # fxmeng/pissa-dataset has a single config called "default"
    dataset = load_dataset(hf_path, split=split, **load_kwargs)

    # Filter by sub_task
    if sub_task is not None and sub_task.lower() not in ("all", "none", ""):
        sub_task_lower = sub_task.lower()
        if sub_task_lower in _PISSA_SUB_TASK_TYPES:
            allowed_types = _PISSA_SUB_TASK_TYPES[sub_task_lower]
            logger.info("Filtering pissa-dataset for sub_task='%s' (types: %s)", sub_task, allowed_types)
            dataset = dataset.filter(
                lambda ex: ex["type"] in allowed_types,
                num_proc=1,
                desc=f"Filtering sub_task={sub_task}",
            )
        else:
            # Treat sub_task as a literal type value
            logger.info("Filtering pissa-dataset for literal type='%s'", sub_task)
            dataset = dataset.filter(
                lambda ex: ex["type"] == sub_task,
                num_proc=1,
                desc=f"Filtering type={sub_task}",
            )

    logger.info("After sub_task filter: %d examples.", len(dataset))

    # Truncate
    if max_samples is not None and max_samples > 0:
        n = min(max_samples, len(dataset))
        dataset = dataset.select(range(n))
        logger.info("Truncated to %d samples.", n)

    # Map columns to standard schema
    dataset = dataset.map(
        _pissa_to_standard,
        batched=False,
        remove_columns=dataset.column_names,
        desc="Mapping pissa-dataset columns",
    )

    logger.info("Loaded %d examples from fxmeng/pissa-dataset (sub_task=%s).", len(dataset), sub_task)
    return dataset


def load_metamathqa(
    hf_path: str = "meta-math/MetaMathQA",
    subset: Optional[str] = None,
    split: str = "train",
    max_samples: Optional[int] = None,
    hf_token: Optional[str] = None,
) -> Dataset:
    """Load MetaMathQA dataset from Hugging Face.

    This function also handles datasets that use 'instruction'/'output' columns
    (like fxmeng/pissa-dataset) by mapping them to 'query'/'response'.

    Args:
        hf_path: HF dataset path.
        subset: Dataset subset/config name (None for default).
        split: Dataset split ('train', 'validation', etc.).
        max_samples: Maximum number of samples to load.
        hf_token: HuggingFace authentication token.

    Returns:
        HuggingFace Dataset object with 'query' and 'response' columns.
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

    # Normalize column names to query/response if needed
    cols = dataset.column_names
    if "query" not in cols and "instruction" in cols:
        logger.info("Mapping instruction/output columns to query/response schema.")
        dataset = dataset.map(
            _pissa_to_standard,
            batched=False,
            remove_columns=dataset.column_names,
            desc="Column mapping",
        )

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
    prompt_template: str = "alpaca",
    num_proc: int = 4,
) -> Dataset:
    """Tokenize the dataset for causal language modeling.

    Applies template, tokenizes, creates labels (same as input_ids for CLM).

    Args:
        dataset: Raw dataset with 'query' and 'response' fields.
        tokenizer: HuggingFace tokenizer with padding_side='right'.
        max_seq_length: Maximum sequence length (truncates longer sequences).
        prompt_template: Template name ('alpaca' or 'alpaca_math' — both use Alpaca format).
        num_proc: Number of processes for parallel tokenization.

    Returns:
        Tokenized dataset with 'input_ids', 'attention_mask', 'labels'.
    """
    # Accept both "alpaca" and "alpaca_math" (legacy alias)
    if prompt_template not in ("alpaca", "alpaca_math"):
        raise ValueError(
            f"Unsupported template: '{prompt_template}'. "
            "Supported: 'alpaca', 'alpaca_math'"
        )

    def tokenize_fn(example: dict) -> dict:
        full_text = format_alpaca(example)

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
