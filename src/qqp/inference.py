"""Load a trained model and classify question pairs."""

from dataclasses import dataclass
from pathlib import Path

from qqp.config import SbertConfig
from qqp.models import SbertSimilarity, SiameseClassifier, TfidfSimilarity
from qqp.utils import read_json

MODEL_SLUGS = ("tfidf", "siamese_lstm", "siamese_bilstm", "sbert", "sbert_finetuned")


@dataclass(frozen=True)
class Prediction:
    model: str
    score: float
    threshold: float

    @property
    def is_duplicate(self) -> bool:
        return self.score >= self.threshold

    @property
    def label(self) -> str:
        return "Duplicate" if self.is_duplicate else "Not duplicate"


class DuplicateDetector:
    def __init__(self, scorer, threshold: float, name: str):
        self.scorer = scorer
        self.threshold = threshold
        self.name = name

    def predict(self, question1: str, question2: str) -> Prediction:
        score = float(self.scorer.score([question1], [question2])[0])
        return Prediction(self.name, score, self.threshold)


def available_models(artifacts_dir: str | Path = "artifacts") -> list[str]:
    """Slugs of every trained model found on disk, simplest model first."""
    root = Path(artifacts_dir)
    return [slug for slug in MODEL_SLUGS if (root / slug / "meta.json").exists()]


def best_model_slug(results_dir: str | Path = "results") -> str:
    summary = Path(results_dir) / "summary.json"
    if not summary.exists():
        raise FileNotFoundError(f"{summary} not found. Run `qqp report` first.")
    return read_json(summary)["best_model"]


def load_detector(
    slug: str = "best",
    artifacts_dir: str | Path = "artifacts",
    results_dir: str | Path = "results",
    device: str | None = None,
) -> DuplicateDetector:
    if slug == "best":
        slug = best_model_slug(results_dir)
    directory = Path(artifacts_dir) / slug
    if not (directory / "meta.json").exists():
        raise FileNotFoundError(f"No trained model at {directory}. Run `qqp train {slug}` first.")
    meta = read_json(directory / "meta.json")

    if slug == "tfidf":
        scorer = TfidfSimilarity.load(directory)
    elif slug in ("siamese_lstm", "siamese_bilstm"):
        scorer = SiameseClassifier.load(directory, device=device)
    elif slug == "sbert":
        config = meta["config"]
        scorer = SbertSimilarity(
            config["model_name"],
            cfg=SbertConfig(max_seq_length=config["max_seq_length"]),
            device=device,
        )
    elif slug == "sbert_finetuned":
        scorer = SbertSimilarity.load(
            directory,
            cfg=SbertConfig(max_seq_length=meta["config"]["max_seq_length"]),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model '{slug}'")
    return DuplicateDetector(scorer, meta["threshold"], meta["name"])
