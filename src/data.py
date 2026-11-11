"""
Dataset loading, tokenization, and balanced sampling for sequence classification tasks.
"""

from typing import Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from src.config import ExperimentConfig


def get_tokenizer(model_id: str) -> PreTrainedTokenizer:
    """Instantiates and returns the pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(model_id)


def preprocess_batch(batch: dict, tokenizer: PreTrainedTokenizer, max_length: int = 128) -> dict:
    """Tokenizes text batches with fixed truncation and max_length padding."""
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def load_and_preprocess_task_a(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """Load AG News dataset and tokenize."""
    try:
        raw_dataset = load_dataset(config.dataset_a_id)
    except Exception:
        raw_dataset = load_dataset("ag_news")

    def _tokenize(batch):
        return preprocess_batch(batch, tokenizer, config.max_length)

    tokenized = raw_dataset.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Shuffle before subsetting
    shuffled_train = tokenized["train"].shuffle(seed=config.seed)
    shuffled_test = tokenized["test"].shuffle(seed=config.seed)

    train_subset = shuffled_train.select(range(config.train_size_a))
    test_subset = shuffled_test.select(range(config.test_size_a))

    return train_subset, test_subset


def load_and_preprocess_task_b(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizer
) -> Tuple[Dataset, Dataset]:
    """Load IMDB sentiment dataset and tokenize."""
    try:
        raw_dataset = load_dataset(config.dataset_b_id)
    except Exception:
        raw_dataset = load_dataset("imdb")

    def _tokenize(batch):
        return preprocess_batch(batch, tokenizer, config.max_length)

    tokenized = raw_dataset.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Shuffle before subsetting so we get a balanced mix of classes
    # (IMDB is sorted by label: first half negative, second half positive)
    shuffled_train = tokenized["train"].shuffle(seed=config.seed)
    shuffled_test = tokenized["test"].shuffle(seed=config.seed)

    train_subset = shuffled_train.select(range(config.train_size_b))
    test_subset = shuffled_test.select(range(config.test_size_b))

    return train_subset, test_subset
