"""
Configuration dataclasses for fine-tuning and continual learning experiments.
"""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class ExperimentConfig:
    # Model & Datasets
    model_id: str = "distilbert-base-uncased"
    dataset_a_id: str = "fancyzhx/ag_news"
    dataset_b_id: str = "stanfordnlp/imdb"
    num_labels_a: int = 4
    num_labels_b: int = 2
    max_length: int = 128

    # Subsets & Reproducibility
    seed: int = 42
    train_size_a: int = 2000
    test_size_a: int = 500
    train_size_b: int = 2000
    test_size_b: int = 500

    # Training common
    batch_size: int = 16
    epochs: int = 3
    warmup_ratio: float = 0.06
    logging_steps: int = 50

    # Full Fine-Tuning hyperparameters
    learning_rate_ft: float = 2e-5
    weight_decay_ft: float = 0.01

    # LoRA / PEFT hyperparameters
    lora_ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    # Constant scaling factor: alpha / r = lora_scaling_ratio
    lora_scaling_ratio: float = 2.0
    lora_dropout: float = 0.05
    learning_rate_lora: float = 1e-4
    weight_decay_lora: float = 0.01
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_lin", "v_lin"])

    # Hardware & Paths
    output_dir_base: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def get_lora_alpha(self, r: int) -> int:
        """Keep alpha / r scaling ratio constant."""
        return int(r * self.lora_scaling_ratio)
