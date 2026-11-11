#!/usr/bin/env python3
"""
Script to run the fine-tuning and continual learning experiments.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ExperimentConfig
from src.utils import set_seed
from src.data import get_tokenizer, load_and_preprocess_task_a, load_and_preprocess_task_b
from src.training import run_full_fine_tuning, run_lora_tradeoff_experiment, select_optimal_lora
from src.continual import run_catastrophic_forgetting_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per task")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for reports & figures")
    return parser.parse_args()


def plot_results(df_tradeoff: pd.DataFrame, df_cf: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Figure 1: Accuracy vs Trainable Parameters (%)
    plt.figure(figsize=(9, 5))
    full_acc = df_tradeoff[df_tradeoff["Method"] == "Full Fine-Tuning"]["Accuracy"].iloc[0]

    palette = sns.color_palette("viridis", len(df_tradeoff))
    scatter = plt.scatter(
        df_tradeoff["% Trainable"],
        df_tradeoff["Accuracy"],
        s=120,
        c=df_tradeoff["Time (s)"],
        cmap="viridis",
        edgecolor="black",
        linewidth=1.2,
        zorder=3
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Training Time (seconds)", fontsize=11)

    for _, row in df_tradeoff.iterrows():
        label = "Full FT" if row["Method"] == "Full Fine-Tuning" else f"LoRA r={row['r']}"
        offset_x = 1.0 if row["% Trainable"] < 50 else -7.0
        plt.annotate(
            label,
            (row["% Trainable"] + offset_x, row["Accuracy"]),
            fontsize=10,
            fontweight="bold"
        )

    plt.axhline(full_acc, color="crimson", linestyle="--", linewidth=1.5, label=f"Full FT Ceiling ({full_acc:.4f})")
    plt.title("Performance vs. Parameter Efficiency Frontier (Task A: AG News)", fontsize=13, fontweight="bold")
    plt.xlabel("Trainable Parameters (% of Full Model)", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim(0.80, 0.92)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tradeoff_frontier.png", dpi=300)
    plt.close()

    # Figure 2: Catastrophic Forgetting Degradation (%)
    plt.figure(figsize=(7, 5))
    colors = ["#2ecc71" if "LoRA" in m else "#e74c3c" for m in df_cf["Method"]]
    bars = plt.bar(df_cf["Method"], df_cf["Loss (%)"], color=colors, edgecolor="black", width=0.5, zorder=3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 1.5,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.title("Catastrophic Forgetting Severity (Accuracy Loss on Task A)", fontsize=13, fontweight="bold")
    plt.ylabel("Performance Degradation (%)", fontsize=11)
    plt.ylim(0, max(df_cf["Loss (%)"]) + 15)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/catastrophic_forgetting.png", dpi=300)
    plt.close()

    print(f"\n[Artifacts] Visualizations successfully saved to '{output_dir}/'")


def main():
    args = parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )

    print(f"Device: {config.device.upper()} | Seed: {config.seed} | Epochs: {config.epochs}")
    set_seed(config.seed)

    # Load Data
    print("\n--- Loading & Preprocessing Datasets ---")
    tokenizer = get_tokenizer(config.model_id)
    train_a, test_a = load_and_preprocess_task_a(config, tokenizer)
    train_b, test_b = load_and_preprocess_task_b(config, tokenizer)

    print(f"Task A (AG News): Train {len(train_a)} | Test {len(test_a)}")
    print(f"Task B (IMDB):    Train {len(train_b)} | Test {len(test_b)} (Balanced 50/50)")

    # Baseline: Full Fine-Tuning
    ft_metrics, model_full = run_full_fine_tuning(train_a, test_a, config)
    full_trainable_params = ft_metrics["Trainable Parameters"]

    # LoRA Trade-off Search
    df_lora, trained_lora_models = run_lora_tradeoff_experiment(
        train_a, test_a, config, full_trainable_params
    )

    # Combine Results
    df_tradeoff = pd.concat([pd.DataFrame([ft_metrics]), df_lora], ignore_index=True)
    print("\n=== Trade-off Results Summary ===")
    print(df_tradeoff.to_string(index=False))

    # 4. Select Optimal LoRA Rank
    optimal_r = select_optimal_lora(df_lora)
    optimal_lora_model = trained_lora_models[optimal_r]

    # Continual Learning / Catastrophic Forgetting Experiment
    cf_models = {
        f"LoRA (r={optimal_r})": optimal_lora_model,
        "Full Fine-Tuning": model_full
    }
    initial_accs = {
        f"LoRA (r={optimal_r})": df_lora[df_lora["r"] == optimal_r]["Accuracy"].iloc[0],
        "Full Fine-Tuning": ft_metrics["Accuracy"]
    }

    df_cf = run_catastrophic_forgetting_experiment(
        models_dict=cf_models,
        initial_accuracies_task_a=initial_accs,
        train_dataset_b=train_b,
        test_dataset_b=test_b,
        test_dataset_a=test_a,
        config=config
    )

    print("\n=== Catastrophic Forgetting Summary ===")
    print(df_cf.to_string(index=False))

    # Save Artifacts & Figures
    os.makedirs(args.output_dir, exist_ok=True)
    df_tradeoff.to_csv(f"{args.output_dir}/results_tradeoff.csv", index=False)
    df_cf.to_csv(f"{args.output_dir}/results_catastrophic_forgetting.csv", index=False)
    plot_results(df_tradeoff, df_cf, args.output_dir)

    print("\n[Done] Benchmark execution completed successfully!")


if __name__ == "__main__":
    main()
