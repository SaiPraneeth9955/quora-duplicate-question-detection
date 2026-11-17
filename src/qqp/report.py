"""Build the comparison table and figures from saved results."""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

from qqp.pipeline import MODELS
from qqp.utils import read_json, write_json

log = logging.getLogger(__name__)

METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
}

# Each model keeps one colour across every figure.
MODEL_COLORS = {
    "tfidf": "#2a78d6",
    "siamese_lstm": "#eb6834",
    "siamese_bilstm": "#1baf7a",
    "sbert": "#eda100",
    "sbert_finetuned": "#e87ba4",
}
TRAIN_COLOR, TUNE_COLOR = "#2a78d6", "#eb6834"
INK, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blues", ["#eef4fd", "#86b6ef", "#256abf", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def load_results(results_dir: str | Path) -> list[dict]:
    """Saved results for every trained model, in canonical order."""
    root = Path(results_dir)
    return [
        read_json(root / slug / "metrics.json")
        for slug in MODELS
        if (root / slug / "metrics.json").exists()
    ]


def comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"Model": result["name"]}
        row.update({label: result["metrics"][key] for key, label in METRICS.items()})
        row["Threshold"] = result["threshold"]
        row["Train time (min)"] = result["train_seconds"] / 60
        row["Inference (pairs/s)"] = result["n_validation"] / result["validation_inference_seconds"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def to_markdown(table: pd.DataFrame) -> str:
    def cell(column: str, value: float) -> str:
        if column == "Train time (min)":
            return "-" if value == 0 else f"{value:.1f}"
        if column == "Inference (pairs/s)":
            return f"{value:,.0f}"
        if column == "Threshold":
            return f"{value:.3f}"
        return f"{value:.4f}"

    header = "| Model | " + " | ".join(table.columns) + " |"
    divider = "|---|" + "---:|" * len(table.columns)
    lines = [
        f"| {model} | " + " | ".join(cell(c, row[c]) for c in table.columns) + " |"
        for model, row in table.iterrows()
    ]
    return "\n".join([header, divider, *lines]) + "\n"


def plot_metrics(results: list[dict], path: Path) -> None:
    names = [r["name"] for r in results]
    colors = [MODEL_COLORS[r["slug"]] for r in results]
    positions = np.arange(len(results))[::-1]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 0.5 * len(results) + 1.4), sharey=True)
    for ax, (key, label) in zip(axes, METRICS.items(), strict=True):
        values = [r["metrics"][key] for r in results]
        ax.barh(positions, values, height=0.62, color=colors)
        for position, value in zip(positions, values, strict=True):
            ax.text(value + 0.02, position, f"{value:.3f}", va="center", color=INK, fontsize=9)
        ax.set_title(label)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    axes[0].set_yticks(positions, names)
    fig.suptitle("Validation metrics by model", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(results: list[dict], results_dir: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for result in results:
        saved = np.load(results_dir / result["slug"] / "scores.npz")
        fpr, tpr, _ = roc_curve(saved["labels"], saved["scores"])
        ax.plot(
            fpr,
            tpr,
            color=MODEL_COLORS[result["slug"]],
            label=f"{result['name']}  (AUC {result['metrics']['roc_auc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the validation set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(results: list[dict], path: Path) -> None:
    classes = ["Not duplicate", "Duplicate"]
    fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.7), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        counts = np.array(result["metrics"]["confusion_matrix"])
        shares = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for (row, col), count in np.ndenumerate(counts):
            ax.text(
                col,
                row,
                f"{count:,}\n{shares[row, col]:.1%}",
                ha="center",
                va="center",
                fontsize=9,
                color="#ffffff" if shares[row, col] > 0.5 else INK,
            )
        ax.set_xticks([0, 1], classes)
        ax.set_yticks([0, 1], classes, rotation=90, va="center")
        ax.set_xlabel("Predicted")
        ax.set_title(result["name"], fontsize=10)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0][0].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rnn_history(name: str, history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    best = min(history, key=lambda h: h["tune_loss"])["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, key, label in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(
            epochs,
            [h[f"train_{key}"] for h in history],
            color=TRAIN_COLOR,
            marker="o",
            label="Train",
        )
        ax.plot(
            epochs,
            [h[f"tune_{key}"] for h in history],
            color=TUNE_COLOR,
            marker="o",
            label="Tuning split",
        )
        ax.axvline(best, color=AXIS, linewidth=1, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_xticks(epochs)
        ax.set_title(f"{name}: {label.lower()}")
    axes[0].annotate(
        "restored checkpoint",
        xy=(best, axes[0].get_ylim()[1]),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sbert_history(history: list[dict], path: Path) -> None:
    trained = [h for h in history if h["train_loss"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(
        [h["step"] for h in trained],
        [h["train_loss"] for h in trained],
        color=TRAIN_COLOR,
        marker="o",
    )
    axes[0].set_title("Fine-tuned Sentence-BERT: training loss")
    axes[0].set_ylabel("Online contrastive loss per pair")
    axes[1].plot(
        [h["step"] for h in history], [h["tune_f1"] for h in history], color=TUNE_COLOR, marker="o"
    )
    axes[1].set_title("Fine-tuned Sentence-BERT: tuning-split F1")
    for ax in axes:
        ax.set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    results = load_results(results_dir)
    if not results:
        raise FileNotFoundError(f"No results under {results_dir}. Run `qqp train all` first.")

    table = comparison_table(results)
    table.to_csv(results_dir / "comparison.csv", float_format="%.4f")
    (results_dir / "comparison.md").write_text(to_markdown(table), encoding="utf-8")

    ranked = sorted(
        results, key=lambda r: (r["metrics"]["f1"], r["metrics"]["roc_auc"]), reverse=True
    )
    write_json(
        results_dir / "summary.json",
        {
            "best_model": ranked[0]["slug"],
            "selection_rule": "highest validation F1, ties broken by ROC-AUC",
            "ranking": [r["slug"] for r in ranked],
        },
    )

    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_metrics(results, figures / "metrics.png")
    plot_roc(results, results_dir, figures / "roc_curves.png")
    plot_confusion_matrices(results, figures / "confusion_matrices.png")
    for result in results:
        history_path = results_dir / result["slug"] / "history.json"
        if not history_path.exists():
            continue
        history = read_json(history_path)
        target = figures / f"{result['slug']}_training.png"
        if result["slug"] == "sbert_finetuned":
            plot_sbert_history(history, target)
        else:
            plot_rnn_history(result["name"], history, target)

    log.info("best model: %s", ranked[0]["name"])
    log.info("report written to %s", results_dir)
    return table
