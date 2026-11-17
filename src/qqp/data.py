"""Dataset loading, text normalisation and the train / tune / validation protocol."""

import logging
import re
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from qqp.config import DataConfig

log = logging.getLogger(__name__)

COLUMNS = ["question1", "question2", "label"]

_APOSTROPHES = re.compile(r"['’]")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation and collapse whitespace.

    Used by the TF-IDF and recurrent models. Sentence-BERT receives the raw
    text because its tokenizer was trained on unnormalised input.
    """
    text = _APOSTROPHES.sub("", str(text).lower())
    return " ".join(_NON_ALPHANUMERIC.sub(" ", text).split())


@dataclass
class Splits:
    """The three partitions every model sees.

    train       fits model parameters.
    tune        held out from the GLUE training set; drives early stopping and
                threshold selection.
    validation  the official GLUE validation set; used only for reported metrics.
    """

    train: pd.DataFrame
    tune: pd.DataFrame
    validation: pd.DataFrame


def load_splits(cfg: DataConfig, seed: int) -> Splits:
    raw = load_dataset(cfg.dataset, cfg.subset)
    train_full = raw["train"].to_pandas()[COLUMNS]
    validation = raw["validation"].to_pandas()[COLUMNS].reset_index(drop=True)

    if cfg.subsample is not None and cfg.subsample < len(train_full):
        train_full = train_full.sample(cfg.subsample, random_state=seed)

    train, tune = train_test_split(
        train_full,
        test_size=cfg.tune_fraction,
        stratify=train_full["label"],
        random_state=seed,
    )
    splits = Splits(train.reset_index(drop=True), tune.reset_index(drop=True), validation)
    for name in ("train", "tune", "validation"):
        frame = getattr(splits, name)
        log.info(
            "%-10s %7d pairs, %.1f%% duplicates", name, len(frame), 100 * frame["label"].mean()
        )
    return splits
