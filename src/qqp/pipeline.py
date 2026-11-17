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
from qqp.utils import resolve_device, write_json

log = logging.getLogger(__name__)


@dataclass
class Trained:
    scorer: object
    train_seconds: float
    config: dict
    history: list[dict] | None = None
    save: bool = True


def _train_tfidf(cfg: Config, splits: Splits, device: str) -> Trained:
    started = time.perf_counter()
    corpus = pd.concat([splits.train["question1"], splits.train["question2"]])
    scorer = TfidfSimilarity(cfg.tfidf).fit(corpus)
    return Trained(scorer, time.perf_counter() - started, asdict(cfg.tfidf))


def _train_siamese(bidirectional: bool) -> Callable[[Config, Splits, str], Trained]:
    def train(cfg: Config, splits: Splits, device: str) -> Trained:
        scorer = SiameseClassifier(cfg.rnn, bidirectional=bidirectional, device=device)
        started = time.perf_counter()
        history = scorer.fit(splits.train, splits.tune, seed=cfg.seed)
        return Trained(scorer, time.perf_counter() - started, asdict(cfg.rnn), history)

    return train


def _train_sbert(cfg: Config, splits: Splits, device: str) -> Trained:
    # Pretrained weights are used unchanged; there is nothing to fit or save.
    scorer = SbertSimilarity(cfg=cfg.sbert, device=device)
    config = {"model_name": cfg.sbert.model_name, "max_seq_length": cfg.sbert.max_seq_length}
    return Trained(scorer, 0.0, config, save=False)


def _train_sbert_finetuned(cfg: Config, splits: Splits, device: str) -> Trained:
    scorer = SbertSimilarity(cfg=cfg.sbert, device=device)
    started = time.perf_counter()
    history = scorer.finetune(splits.train, splits.tune, seed=cfg.seed)
    return Trained(scorer, time.perf_counter() - started, asdict(cfg.sbert), history)


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    name: str
    train: Callable[[Config, Splits, str], Trained]


MODELS: dict[str, ModelSpec] = {
    spec.slug: spec
    for spec in (
        ModelSpec("tfidf", "TF-IDF + cosine", _train_tfidf),
        ModelSpec("siamese_lstm", "Siamese LSTM", _train_siamese(bidirectional=False)),
        ModelSpec("siamese_bilstm", "Siamese BiLSTM", _train_siamese(bidirectional=True)),
        ModelSpec("sbert", "Sentence-BERT + cosine", _train_sbert),
        ModelSpec("sbert_finetuned", "Fine-tuned Sentence-BERT", _train_sbert_finetuned),
    )
}


def run_model(slug: str, cfg: Config, splits: Splits, device: str | None = None) -> dict:
    """Train one model, pick its threshold on the tuning split and score validation."""
    spec = MODELS[slug]
    device = resolve_device(device)
    log.info("=== %s ===", spec.name)

    trained = spec.train(cfg, splits, device)
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
        "threshold": threshold,
        "metrics": classification_metrics(labels, scores, threshold),
        "train_seconds": trained.train_seconds,
        "validation_inference_seconds": inference_seconds,
        "n_train": len(splits.train),
        "n_tune": len(splits.tune),
        "n_validation": len(splits.validation),
        "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "seed": cfg.seed,
        "config": trained.config,
    }

    results_dir = Path(cfg.results_dir) / spec.slug
    write_json(results_dir / "metrics.json", result)
    np.savez_compressed(results_dir / "scores.npz", scores=scores, labels=labels)
    if trained.history:
        write_json(results_dir / "history.json", trained.history)

    artifact_dir = Path(cfg.artifacts_dir) / spec.slug
    if trained.save:
        scorer.save(artifact_dir)
    write_json(
        artifact_dir / "meta.json",
        {"slug": spec.slug, "name": spec.name, "threshold": threshold, "config": trained.config},
    )

    metrics = result["metrics"]
    log.info(
        "%s: accuracy %.4f  precision %.4f  recall %.4f  F1 %.4f  ROC-AUC %.4f  (threshold %.3f)",
        spec.name,
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
