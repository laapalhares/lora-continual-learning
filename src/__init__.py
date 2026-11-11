"""Source modules for fine-tuning and continual learning experiments."""

from src.config import ExperimentConfig
from src.utils import set_seed, count_parameters
from src.metrics import compute_classification_metrics
from src.data import get_tokenizer, load_and_preprocess_task_a, load_and_preprocess_task_b
from src.models import get_base_model, apply_lora
from src.training import run_full_fine_tuning, run_lora_tradeoff_experiment, select_optimal_lora
from src.continual import run_catastrophic_forgetting_experiment

__all__ = [
    "ExperimentConfig",
    "set_seed",
    "count_parameters",
    "compute_classification_metrics",
    "get_tokenizer",
    "load_and_preprocess_task_a",
    "load_and_preprocess_task_b",
    "get_base_model",
    "apply_lora",
    "run_full_fine_tuning",
    "run_lora_tradeoff_experiment",
    "select_optimal_lora",
    "run_catastrophic_forgetting_experiment",
]
