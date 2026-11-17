"""Train, evaluate and persist each model under the shared protocol."""

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qqp.config import Config
from qqp.data import Splits
from qqp.metrics import best_f1_threshold, classification_metrics
from qqp.models import SbertSimilarity, SiameseClassifier, TfidfSimilarity
from qqp.utils import read_json, resolve_device, write_json

log = logging.getLogger(__name__)


@dataclass
class Trained:
    scorer: object
    train_seconds: float
    config: dict
    history: list[dict] | None = None
    save: bool = True


def _train_tfidf(cfg: Config, splits: Splits, device: str, seed: int | None) -> Trained:
    started = time.perf_counter()
    corpus = pd.concat([splits.train["question1"], splits.train["question2"]])
    scorer = TfidfSimilarity(cfg.tfidf).fit(corpus)
    return Trained(scorer, time.perf_counter() - started, asdict(cfg.tfidf))


def _train_siamese(bidirectional: bool) -> Callable[[Config, Splits, str, int | None], Trained]:
    def train(cfg: Config, splits: Splits, device: str, seed: int | None) -> Trained:
        scorer = SiameseClassifier(cfg.rnn, bidirectional=bidirectional, device=device)
        started = time.perf_counter()
        history = scorer.fit(splits.train, splits.tune, seed=seed)
        return Trained(scorer, time.perf_counter() - started, asdict(cfg.rnn), history)

    return train


def _train_sbert(cfg: Config, splits: Splits, device: str, seed: int | None) -> Trained:
    # Pretrained weights are used unchanged; there is nothing to fit or save.
    scorer = SbertSimilarity(cfg=cfg.sbert, device=device)
    config = {"model_name": cfg.sbert.model_name, "max_seq_length": cfg.sbert.max_seq_length}
    return Trained(scorer, 0.0, config, save=False)


def _train_sbert_finetuned(cfg: Config, splits: Splits, device: str, seed: int | None) -> Trained:
    scorer = SbertSimilarity(cfg=cfg.sbert, device=device)
    started = time.perf_counter()
    history = scorer.finetune(splits.train, splits.tune, seed=seed)
    return Trained(scorer, time.perf_counter() - started, asdict(cfg.sbert), history)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    name: str
    train: Callable[[Config, Splits, str, int | None], Trained]
    # Stochastic models depend on the training seed and are run once per seed.
    stochastic: bool


MODELS: dict[str, ModelSpec] = {
    spec.slug: spec
    for spec in (
        ModelSpec("tfidf", "TF-IDF + cosine", _train_tfidf, stochastic=False),
        ModelSpec("siamese_lstm", "Siamese LSTM", _train_siamese(False), stochastic=True),
        ModelSpec("siamese_bilstm", "Siamese BiLSTM", _train_siamese(True), stochastic=True),
        ModelSpec("sbert", "Sentence-BERT + cosine", _train_sbert, stochastic=False),
        ModelSpec(
            "sbert_finetuned", "Fine-tuned Sentence-BERT", _train_sbert_finetuned, stochastic=True
        ),
    )
}


def run_name(seed: int | None) -> str:
    """Directory name of one run inside ``results/<model>/``."""
    return "deterministic" if seed is None else f"seed_{seed}"


def _is_current(run_dir: Path, config: dict, splits: Splits) -> bool:
    """True when a finished run with the same settings and data is already on disk."""
    path = run_dir / "metrics.json"
    if not path.exists() or not (run_dir / "scores.npz").exists():
        return False
    saved = read_json(path)
    return (
        saved["config"] == config
        and saved["n_train"] == len(splits.train)
        and saved["n_tune"] == len(splits.tune)
    )


def _expected_config(spec: ModelSpec, cfg: Config) -> dict:
    if spec.slug == "tfidf":
        return asdict(cfg.tfidf)
    if spec.slug == "sbert":
        return {"model_name": cfg.sbert.model_name, "max_seq_length": cfg.sbert.max_seq_length}
    if spec.slug == "sbert_finetuned":
        return asdict(cfg.sbert)
    return asdict(cfg.rnn)


def _run_once(
    spec: ModelSpec,
    cfg: Config,
    splits: Splits,
    device: str,
    seed: int | None,
    save_artifact: bool,
) -> dict:
    label = spec.name if seed is None else f"{spec.name} (seed {seed})"
    log.info("=== %s ===", label)

    trained = spec.train(cfg, splits, device, seed)
    scorer = trained.scorer

    tune_labels = splits.tune["label"].to_numpy()
    threshold = best_f1_threshold(
        tune_labels, scorer.score(splits.tune["question1"], splits.tune["question2"])
    )

    started = time.perf_counter()
    scores = scorer.score(splits.validation["question1"], splits.validation["question2"])
    inference_seconds = time.perf_counter() - started
    labels = splits.validation["label"].to_numpy()

    result = {
        "slug": spec.slug,
        "name": spec.name,
        "seed": seed,
        "saved_for_inference": save_artifact,
        "threshold": threshold,
        "metrics": classification_metrics(labels, scores, threshold),
        "train_seconds": trained.train_seconds,
        "validation_inference_seconds": inference_seconds,
        "n_train": len(splits.train),
        "n_tune": len(splits.tune),
        "n_validation": len(splits.validation),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "split_seed": cfg.split_seed,
        "config": trained.config,
    }

    if save_artifact:
        artifact_dir = Path(cfg.artifacts_dir) / spec.slug
        if trained.save:
            scorer.save(artifact_dir)
        write_json(
            artifact_dir / "meta.json",
            {
                "slug": spec.slug,
                "name": spec.name,
                "seed": seed,
                "threshold": threshold,
                "config": trained.config,
            },
        )

    # metrics.json is written last: its presence marks the run as complete.
    run_dir = Path(cfg.results_dir) / spec.slug / run_name(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "scores.npz", scores=scores, labels=labels)
    if trained.history:
        write_json(run_dir / "history.json", trained.history)
    write_json(run_dir / "metrics.json", result)

    metrics = result["metrics"]
    log.info(
        "%s: accuracy %.4f  precision %.4f  recall %.4f  F1 %.4f  ROC-AUC %.4f  (threshold %.3f)",
        label,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        threshold,
    )

    del trained, scorer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_model(
    slug: str,
    cfg: Config,
    splits: Splits,
    device: str | None = None,
    force: bool = False,
) -> list[dict]:
    """Train and evaluate one model, once per seed if its training is stochastic.

    Each run picks its threshold on the tuning split and is then scored on
    validation. The model from the first seed is the one saved for inference.
    Runs already on disk with identical settings are reused unless ``force``.
    """
    spec = MODELS[slug]
    device = resolve_device(device)
    seeds: list[int | None] = list(cfg.seeds) if spec.stochastic else [None]
    expected = _expected_config(spec, cfg)

    results = []
    for position, seed in enumerate(seeds):
        primary = position == 0
        run_dir = Path(cfg.results_dir) / spec.slug / run_name(seed)
        meta_path = Path(cfg.artifacts_dir) / spec.slug / "meta.json"
        has_artifact = meta_path.exists() and all(
            read_json(meta_path).get(key) == value
            for key, value in (("config", expected), ("seed", seed))
        )
        if not force and _is_current(run_dir, expected, splits) and (has_artifact or not primary):
            log.info("%s %s: already complete, reusing saved run", spec.name, run_name(seed))
            results.append(read_json(run_dir / "metrics.json"))
            continue
        results.append(_run_once(spec, cfg, splits, device, seed, save_artifact=primary))
    return results
