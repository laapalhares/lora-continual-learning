"""Catastrophic forgetting evaluation by sequential task adaptation."""

import copy
from typing import Dict, Any, List
import torch
import pandas as pd
from transformers import Trainer, TrainingArguments
from peft import PeftModel
from src.config import ExperimentConfig
from src.metrics import compute_classification_metrics


def get_classifier_module(model: torch.nn.Module) -> torch.nn.Module:
    """Get the classification head module."""
    if isinstance(model, PeftModel):
        return model.base_model.model.classifier
    return model.classifier


def save_classifier_head(model: torch.nn.Module) -> torch.nn.Module:
    """Save a copy of the current classification head."""
    target_module = get_classifier_module(model)
    return copy.deepcopy(target_module)


def set_classifier_head(model: torch.nn.Module, head_module: torch.nn.Module, num_labels: int):
    """Set classification head and update num_labels."""
    model.config.num_labels = num_labels
    if hasattr(model, "num_labels"):
        model.num_labels = num_labels

    if isinstance(model, PeftModel):
        model.base_model.model.classifier = head_module
        model.base_model.model.num_labels = num_labels
        model.base_model.model.config.num_labels = num_labels
    else:
        model.classifier = head_module


def run_catastrophic_forgetting_experiment(
    models_dict: Dict[str, torch.nn.Module],
    initial_accuracies_task_a: Dict[str, float],
    train_dataset_b,
    test_dataset_b,
    test_dataset_a,
    config: ExperimentConfig
) -> pd.DataFrame:
    """Train on Task B and evaluate catastrophic forgetting on Task A."""
    results: List[Dict[str, Any]] = []

    for name, model in models_dict.items():
        print(f"\n==========================================")
        print(f"Starting Continual Learning Test: {name}")
        print(f"==========================================")

        model.to(config.device)
        init_acc_a = initial_accuracies_task_a[name]

        # Save Task A classification head
        saved_head_a = save_classifier_head(model)

        #  Attach new classification head for Task B
        hidden_size = model.config.hidden_size
        new_head_b = torch.nn.Linear(hidden_size, config.num_labels_b).to(config.device)
        set_classifier_head(model, new_head_b, config.num_labels_b)

        # Train on Task B (IMDB)
        training_args_b = TrainingArguments(
            output_dir=f"{config.output_dir_base}/cf_train_{name.replace(' ', '_')}",
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate_ft if "Full" in name else config.learning_rate_lora,
            weight_decay=config.weight_decay_ft,
            save_strategy="no",
            logging_steps=config.logging_steps,
            report_to="none"
        )

        trainer_b = Trainer(
            model=model,
            args=training_args_b,
            train_dataset=train_dataset_b,
            eval_dataset=test_dataset_b,
            compute_metrics=compute_classification_metrics
        )

        print(f"Training {name} on Task B ({config.dataset_b_id})...")
        trainer_b.train()

        # Evaluate on Task B
        eval_b = trainer_b.evaluate()
        acc_task_b = eval_b["eval_accuracy"]
        f1_task_b = eval_b["eval_f1_macro"]
        print(f"-> Task B Adaptation Accuracy: {acc_task_b:.4f} | F1-Macro: {f1_task_b:.4f}")

        # Restore Task A classification head
        print(f"-> Restoring Task A ({config.dataset_a_id}) head...")
        set_classifier_head(model, saved_head_a, config.num_labels_a)

        # Evaluate on Task A test set
        eval_args_a = TrainingArguments(
            output_dir=f"{config.output_dir_base}/tmp_eval",
            per_device_eval_batch_size=config.batch_size,
            report_to="none"
        )
        trainer_forgetting = Trainer(
            model=model,
            args=eval_args_a,
            eval_dataset=test_dataset_a,
            compute_metrics=compute_classification_metrics
        )

        eval_final_a = trainer_forgetting.evaluate()
        final_acc_a = eval_final_a["eval_accuracy"]
        final_f1_a = eval_final_a["eval_f1_macro"]

        # Compute Forgetting
        abs_forgetting = init_acc_a - final_acc_a
        pct_forgetting = (abs_forgetting / init_acc_a * 100) if init_acc_a > 0 else 0.0

        print(f"-> Task A Final Accuracy: {final_acc_a:.4f} (Initial: {init_acc_a:.4f})")
        print(f"-> Catastrophic Forgetting: {abs_forgetting:.4f} ({pct_forgetting:.2f}%)")

        results.append({
            "Method": name,
            "Initial Accuracy (Task A)": round(init_acc_a, 4),
            "Final Accuracy (Task A)": round(final_acc_a, 4),
            "Final F1-Macro (Task A)": round(final_f1_a, 4),
            "Adaptation Accuracy (Task B)": round(acc_task_b, 4),
            "Adaptation F1-Macro (Task B)": round(f1_task_b, 4),
            "Absolute Forgetting": round(abs_forgetting, 4),
            "Loss (%)": round(pct_forgetting, 2)
        })

    return pd.DataFrame(results)
