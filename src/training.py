"""Training loops for Full Fine-Tuning and LoRA."""

import time
from typing import Dict, Any, Tuple
import pandas as pd
import torch
from transformers import Trainer, TrainingArguments
from src.config import ExperimentConfig
from src.utils import count_parameters
from src.metrics import compute_classification_metrics
from src.models import get_base_model, apply_lora


def run_full_fine_tuning(
    train_dataset,
    test_dataset,
    config: ExperimentConfig
) -> Tuple[Dict[str, Any], torch.nn.Module]:
    """Run full fine-tuning baseline."""
    print("\n==========================================")
    print("Running Full Fine-Tuning Baseline")
    print("==========================================")

    model = get_base_model(config.model_id, config.num_labels_a, device=config.device)
    param_info = count_parameters(model)

    training_args = TrainingArguments(
        output_dir=f"{config.output_dir_base}/full_ft",
        eval_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate_ft,
        weight_decay=config.weight_decay_ft,
        num_train_epochs=config.epochs,
        logging_steps=config.logging_steps,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_classification_metrics
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    eval_results = trainer.evaluate()

    metrics = {
        "Method": "Full Fine-Tuning",
        "r": "N/A",
        "Accuracy": round(eval_results["eval_accuracy"], 4),
        "F1-Macro": round(eval_results["eval_f1_macro"], 4),
        "Time (s)": round(training_time, 2),
        "Trainable Parameters": param_info["trainable_parameters"],
        "% Trainable": 100.0
    }

    print(f"Full FT Result -> Accuracy: {metrics['Accuracy']} | F1-Macro: {metrics['F1-Macro']} | Time: {metrics['Time (s)']}s")
    return metrics, model


def run_lora_tradeoff_experiment(
    train_dataset,
    test_dataset,
    config: ExperimentConfig,
    full_trainable_params: int
) -> Tuple[pd.DataFrame, Dict[int, torch.nn.Module]]:
    """Run LoRA across ranks [4, 8, 16, 32]."""
    print("\n==========================================")
    print("Running LoRA Rank Trade-off Experiment")
    print("==========================================")

    lora_results = []
    trained_models: Dict[int, torch.nn.Module] = {}

    for r_value in config.lora_ranks:
        alpha = config.get_lora_alpha(r_value)
        print(f"\n--- Training LoRA (r={r_value}, alpha={alpha}, scaling={alpha/r_value:.1f}) ---")

        base_model = get_base_model(config.model_id, config.num_labels_a, device=config.device)
        lora_model = apply_lora(base_model, r_value, config)

        param_info = count_parameters(lora_model)
        pct_of_full = (param_info["trainable_parameters"] / full_trainable_params) * 100

        print(f"Trainable params: {param_info['trainable_parameters']:,} ({pct_of_full:.2f}% of full model)")

        warmup_steps = int((len(train_dataset) // config.batch_size) * config.epochs * config.warmup_ratio)
        training_args = TrainingArguments(
            output_dir=f"{config.output_dir_base}/lora_r{r_value}",
            eval_strategy="epoch",
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            learning_rate=config.learning_rate_lora,
            weight_decay=config.weight_decay_lora,
            num_train_epochs=config.epochs,
            warmup_steps=warmup_steps,
            logging_steps=config.logging_steps,
            save_strategy="no",
            report_to="none"
        )

        trainer = Trainer(
            model=lora_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_classification_metrics
        )

        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time

        eval_results = trainer.evaluate()

        metrics = {
            "Method": "LoRA",
            "r": r_value,
            "Accuracy": round(eval_results["eval_accuracy"], 4),
            "F1-Macro": round(eval_results["eval_f1_macro"], 4),
            "Time (s)": round(training_time, 2),
            "Trainable Parameters": param_info["trainable_parameters"],
            "% Trainable": round(pct_of_full, 2)
        }

        print(f"LoRA (r={r_value}) -> Accuracy: {metrics['Accuracy']} | F1-Macro: {metrics['F1-Macro']} | Time: {metrics['Time (s)']}s")

        lora_results.append(metrics)
        trained_models[r_value] = lora_model

    df = pd.DataFrame(lora_results)
    return df, trained_models


def select_optimal_lora(df_lora: pd.DataFrame) -> int:
    """Select the smallest rank within 0.5% of peak accuracy."""
    max_acc = df_lora["Accuracy"].max()
    threshold = max_acc * 0.995
    candidates = df_lora[df_lora["Accuracy"] >= threshold]
    optimal_row = candidates.sort_values(by="Trainable Parameters", ascending=True).iloc[0]
    optimal_r = int(optimal_row["r"])
    print(f"\n[Selection] Optimal LoRA Rank Selected: r={optimal_r} (Accuracy: {optimal_row['Accuracy']}, % Trainable: {optimal_row['% Trainable']}%)")
    return optimal_r
