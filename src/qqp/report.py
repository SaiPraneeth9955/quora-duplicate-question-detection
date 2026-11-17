"""Build the comparison table and figures from saved results."""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve

from qqp.pipeline import MODELS, run_name
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


@dataclass
class ModelResults:
    """Every saved run of one model; trained models have one run per seed."""

    slug: str
    name: str
    runs: list[dict]

    @property
    def primary(self) -> dict:
        """The run whose model was saved for inference."""
        return next((r for r in self.runs if r.get("saved_for_inference")), self.runs[0])

    def values(self, key: str) -> np.ndarray:
        if key == "train_minutes":
            return np.array([r["train_seconds"] / 60 for r in self.runs])
        if key == "pairs_per_second":
            return np.array(
                [r["n_validation"] / r["validation_inference_seconds"] for r in self.runs]
            )
        return np.array([r["metrics"][key] for r in self.runs])

    def mean(self, key: str) -> float:
        return float(self.values(key).mean())

    def std(self, key: str) -> float | None:
        """Sample standard deviation across seeds; None for a single run."""
        values = self.values(key)
        return float(values.std(ddof=1)) if len(values) > 1 else None


def load_results(results_dir: str | Path) -> list[ModelResults]:
    """Saved runs for every trained model, in canonical order."""
    root = Path(results_dir)
    models = []
    for slug, spec in MODELS.items():
        runs = [read_json(path) for path in sorted((root / slug).glob("*/metrics.json"))]
        if runs:
            runs.sort(key=lambda r: (r["seed"] is not None, r["seed"] or 0))
            models.append(ModelResults(slug, spec.name, runs))
    return models


def runs_table(models: list[ModelResults]) -> pd.DataFrame:
    """One row per individual run."""
    rows = []
    for model in models:
        for run in model.runs:
            row = {"model": model.slug, "seed": run["seed"], "threshold": run["threshold"]}
            row.update({key: run["metrics"][key] for key in METRICS})
            row["train_minutes"] = run["train_seconds"] / 60
            row["pairs_per_second"] = run["n_validation"] / run["validation_inference_seconds"]
            rows.append(row)
    return pd.DataFrame(rows).astype({"seed": "Int64"})


def comparison_table(models: list[ModelResults]) -> pd.DataFrame:
    """Mean and standard deviation of each metric across a model's runs."""
    rows = []
    for model in models:
        row = {"model": model.slug, "name": model.name, "runs": len(model.runs)}
        for key in (*METRICS, "train_minutes", "pairs_per_second"):
            row[f"{key}_mean"] = model.mean(key)
            row[f"{key}_std"] = model.std(key)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(models: list[ModelResults]) -> str:
    def cell(model: ModelResults, key: str) -> str:
        mean, std = model.mean(key), model.std(key)
        if key == "train_minutes":
            if mean == 0:
                return "-"
            if mean < 0.05:
                return "< 0.1"
            return f"{mean:.1f}" if std is None else f"{mean:.1f} ± {std:.1f}"
        if key == "pairs_per_second":
            return f"{mean:,.0f}"
        return f"{mean:.4f}" if std is None else f"{mean:.4f} ± {std:.4f}"

    columns = {**METRICS, "train_minutes": "Train time (min)", "pairs_per_second": "Pairs/s"}
    header = "| Model | Runs | " + " | ".join(columns.values()) + " |"
    divider = "|---|---:|" + "---:|" * len(columns)
    lines = [
        f"| {model.name} | {len(model.runs)} | "
        + " | ".join(cell(model, key) for key in columns)
        + " |"
        for model in models
    ]
    return "\n".join([header, divider, *lines]) + "\n"


def plot_metrics(models: list[ModelResults], path: Path) -> None:
    names = [m.name for m in models]
    colors = [MODEL_COLORS[m.slug] for m in models]
    positions = np.arange(len(models))[::-1]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 0.5 * len(models) + 1.6), sharey=True)
    for ax, (key, label) in zip(axes, METRICS.items(), strict=True):
        means = [m.mean(key) for m in models]
        spreads = [m.std(key) or 0.0 for m in models]
        ax.barh(positions, means, height=0.62, color=colors)
        ax.errorbar(
            means, positions, xerr=spreads, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3
        )
        for position, mean, spread in zip(positions, means, spreads, strict=True):
            ax.text(
                mean + spread + 0.02, position, f"{mean:.3f}", va="center", color=INK, fontsize=9
            )
        ax.set_title(label)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
        ax.tick_params(length=0)
    axes[0].set_yticks(positions, names)
    fig.suptitle("Validation metrics by model", x=0.01, ha="left", fontsize=13, fontweight="bold")
    if any(len(m.runs) > 1 for m in models):
        fig.text(
            0.01,
            0.0,
            "Bars show the mean over training seeds; whiskers show one standard deviation.",
            fontsize=9,
            color=INK_SECONDARY,
        )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _seed_note(models: list[ModelResults]) -> str | None:
    seeds = {m.primary["seed"] for m in models if m.primary["seed"] is not None}
    if not seeds:
        return None
    return "Trained models: the run with seed " + ", ".join(str(s) for s in sorted(seeds)) + "."


def plot_roc(models: list[ModelResults], results_dir: Path, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6))
    for model in models:
        run = model.primary
        saved = np.load(results_dir / model.slug / run_name(run["seed"]) / "scores.npz")
        fpr, tpr, _ = roc_curve(saved["labels"], saved["scores"])
        spread = model.std("roc_auc")
        auc = f"{model.mean('roc_auc'):.3f}" + ("" if spread is None else f" ± {spread:.3f}")
        ax.plot(fpr, tpr, color=MODEL_COLORS[model.slug], label=f"{model.name}  (AUC {auc})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the validation set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.005)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    note = _seed_note(models)
    if note:
        fig.text(0.02, 0.0, f"{note} AUC is the mean over seeds.", fontsize=9, color=INK_SECONDARY)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(models: list[ModelResults], path: Path) -> None:
    classes = ["Not duplicate", "Duplicate"]
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.9), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        result = model.primary
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
        ax.set_title(model.name, fontsize=10)
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0][0].set_ylabel("Actual")
    note = _seed_note(models)
    if note:
        fig.text(0.01, 0.0, note, fontsize=9, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _seed_lines(ax, histories: list[list[dict]], x: str, y: str, color: str, label: str) -> None:
    """One line per seed in a shared colour, so the spread between runs is visible."""
    for position, history in enumerate(histories):
        points = [(h[x], h[y]) for h in history if h[y] is not None]
        ax.plot(
            *zip(*points, strict=True),
            color=color,
            linewidth=1.5,
            alpha=0.75,
            label=label if position == 0 else None,
        )


def _runs_title(title: str, histories: list[list[dict]]) -> str:
    return title if len(histories) == 1 else f"{title} ({len(histories)} seeds)"


def plot_rnn_history(name: str, histories: list[list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, key in ((axes[0], "loss"), (axes[1], "accuracy")):
        _seed_lines(ax, histories, "epoch", f"train_{key}", TRAIN_COLOR, "Train")
        _seed_lines(ax, histories, "epoch", f"tune_{key}", TUNE_COLOR, "Tuning split")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(_runs_title(f"{name}: {key}", histories))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_sbert_history(histories: list[list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    _seed_lines(axes[0], histories, "step", "train_loss", TRAIN_COLOR, "Train")
    axes[0].set_title(_runs_title("Fine-tuned Sentence-BERT: training loss", histories))
    axes[0].set_ylabel("Online contrastive loss per pair")
    _seed_lines(axes[1], histories, "step", "tune_f1", TUNE_COLOR, "Tuning split")
    axes[1].set_title(_runs_title("Fine-tuned Sentence-BERT: tuning-split F1", histories))
    for ax in axes:
        ax.set_xlabel("Training step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_report(results_dir: str | Path) -> str:
    """Write the tables, summary and figures; returns the comparison table as Markdown."""
    results_dir = Path(results_dir)
    models = load_results(results_dir)
    if not models:
        raise FileNotFoundError(f"No results under {results_dir}. Run `qqp train all` first.")

    markdown = to_markdown(models)
    comparison_table(models).to_csv(
        results_dir / "comparison.csv", index=False, float_format="%.4f"
    )
    runs_table(models).to_csv(results_dir / "runs.csv", index=False, float_format="%.4f")
    (results_dir / "comparison.md").write_text(markdown, encoding="utf-8")

    ranked = sorted(models, key=lambda m: (m.mean("f1"), m.mean("roc_auc")), reverse=True)
    write_json(
        results_dir / "summary.json",
        {
            "best_model": ranked[0].slug,
            "selection_rule": "highest mean validation F1 across seeds, ties broken by ROC-AUC",
            "ranking": [m.slug for m in ranked],
            "models": {
                m.slug: {
                    "runs": len(m.runs),
                    "seeds": [r["seed"] for r in m.runs if r["seed"] is not None],
                    **{key: {"mean": m.mean(key), "std": m.std(key)} for key in METRICS},
                }
                for m in models
            },
        },
    )

    figures = results_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_metrics(models, figures / "metrics.png")
    plot_roc(models, results_dir, figures / "roc_curves.png")
    plot_confusion_matrices(models, figures / "confusion_matrices.png")
    for model in models:
        paths = [
            results_dir / model.slug / run_name(r["seed"]) / "history.json" for r in model.runs
        ]
        histories = [read_json(path) for path in paths if path.exists()]
        if not histories:
            continue
        target = figures / f"{model.slug}_training.png"
        if model.slug == "sbert_finetuned":
            plot_sbert_history(histories, target)
        else:
            plot_rnn_history(model.name, histories, target)

    log.info("best model: %s", ranked[0].name)
    log.info("report written to %s", results_dir)
    return markdown
