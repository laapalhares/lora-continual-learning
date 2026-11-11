"""Model helpers for DistilBERT and LoRA."""

from typing import Optional
import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from src.config import ExperimentConfig


def get_base_model(
    model_id: str,
    num_labels: int,
    device: Optional[str] = None
) -> AutoModelForSequenceClassification:
    """Load DistilBERT for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels
    )
    if device:
        model.to(device)
    return model


def apply_lora(
    base_model: torch.nn.Module,
    r: int,
    config: ExperimentConfig
) -> PeftModel:
    """Apply LoRA adapters to the model."""
    alpha = config.get_lora_alpha(r)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )
    lora_model = get_peft_model(base_model, lora_config)
    return lora_model
