import numpy as np
import pytest

from qqp.metrics import classification_metrics
from qqp.pipeline import run_name
from qqp.report import build_report, load_results
from qqp.utils import read_json, write_json

LABELS = np.array([0, 0, 1, 1, 0, 1])


def _write_run(root, slug, seed, scores, primary=False):
    run_dir = root / slug / run_name(seed)
    run_dir.mkdir(parents=True)
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=LABELS)
    write_json(
        run_dir / "metrics.json",
        {
            "slug": slug,
            "seed": seed,
            "saved_for_inference": primary,
            "threshold": 0.5,
            "metrics": classification_metrics(LABELS, scores, 0.5),
            "train_seconds": 60.0,
            "validation_inference_seconds": 2.0,
            "n_validation": len(LABELS),
        },
    )
    if seed is not None:
        write_json(
            run_dir / "history.json",
            [
                {
                    "epoch": epoch,
                    "train_loss": 0.5 / epoch,
                    "tune_loss": 0.6 / epoch,
                    "train_accuracy": 0.7,
                    "tune_accuracy": 0.65,
                }
                for epoch in (1, 2)
            ],
        )


def test_report_aggregates_runs_across_seeds(tmp_path):
    _write_run(tmp_path, "tfidf", None, np.array([0.1, 0.6, 0.7, 0.4, 0.2, 0.9]), primary=True)
    # Seed 7 is perfect (F1 1.0); seed 8 misses one duplicate (F1 0.8).
    _write_run(tmp_path, "siamese_lstm", 8, np.array([0.1, 0.2, 0.9, 0.4, 0.3, 0.8]))
    _write_run(tmp_path, "siamese_lstm", 7, np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7]), primary=True)

    markdown = build_report(tmp_path)

    tfidf, lstm = load_results(tmp_path)
    assert [run["seed"] for run in lstm.runs] == [7, 8]
    assert lstm.primary["seed"] == 7
    assert lstm.mean("f1") == pytest.approx(0.9)
    assert lstm.std("f1") == pytest.approx(np.std([1.0, 0.8], ddof=1))
    assert tfidf.std("f1") is None

    summary = read_json(tmp_path / "summary.json")
    assert summary["best_model"] == "siamese_lstm"
    assert summary["models"]["siamese_lstm"]["seeds"] == [7, 8]
    assert summary["models"]["tfidf"]["f1"]["std"] is None

    assert "0.9000 ± 0.1414" in markdown
    for name in ("comparison.csv", "comparison.md", "runs.csv"):
        assert (tmp_path / name).exists()
    for figure in ("metrics", "roc_curves", "confusion_matrices", "siamese_lstm_training"):
        assert (tmp_path / "figures" / f"{figure}.png").exists()
