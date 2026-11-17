"""TF-IDF bag-of-n-grams with cosine similarity."""

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from qqp.config import TfidfConfig
from qqp.data import normalize_text


class TfidfSimilarity:
    def __init__(self, cfg: TfidfConfig | None = None):
        cfg = cfg or TfidfConfig()
        self.vectorizer = TfidfVectorizer(
            preprocessor=normalize_text,
            token_pattern=r"\S+",
            ngram_range=(1, cfg.ngram_max),
            min_df=cfg.min_df,
            sublinear_tf=cfg.sublinear_tf,
        )

    def fit(self, questions: Sequence[str]) -> "TfidfSimilarity":
        self.vectorizer.fit(questions)
        return self

    def score(self, questions1: Sequence[str], questions2: Sequence[str]) -> np.ndarray:
        a = self.vectorizer.transform(questions1)
        b = self.vectorizer.transform(questions2)
        # Rows are L2-normalised, so the row-wise dot product is the cosine similarity.
        return np.asarray(a.multiply(b).sum(axis=1)).ravel()

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, directory / "vectorizer.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfSimilarity":
        model = cls()
        model.vectorizer = joblib.load(Path(directory) / "vectorizer.joblib")
        return model
