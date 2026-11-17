"""Experiment configuration, loaded from YAML into typed dataclasses."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset: str = "nyu-mll/glue"
    subset: str = "qqp"
    tune_fraction: float = 0.05
    subsample: int | None = None


@dataclass
class TfidfConfig:
    ngram_max: int = 2
    min_df: int = 2
    sublinear_tf: bool = True


@dataclass
class RnnConfig:
    max_vocab: int = 50_000
    min_freq: int = 2
    max_len: int = 30
    embedding_dim: int = 200
    hidden_dim: int = 128
    dropout: float = 0.3
    pooling: str = "max"
    lr: float = 1e-3
    lr_decay: float = 0.5
    batch_size: int = 512
    epochs: int = 15
    patience: int = 3


@dataclass
class SbertConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_seq_length: int = 64
    batch_size: int = 64
    eval_batch_size: int = 256
    lr: float = 2e-5
    epochs: int = 3
    margin: float = 0.5
    warmup_fraction: float = 0.1
    eval_every: int = 1000


@dataclass
class Config:
    seed: int = 42
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    tfidf: TfidfConfig = field(default_factory=TfidfConfig)
    rnn: RnnConfig = field(default_factory=RnnConfig)
    sbert: SbertConfig = field(default_factory=SbertConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build(cls, values: dict, path: str = ""):
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown config keys under '{path or 'root'}': {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        field_type = known[name].type
        if dataclasses.is_dataclass(field_type):
            value = _build(field_type, value or {}, f"{path}{name}.")
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config. Missing keys fall back to the defaults above."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as handle:
        return _build(Config, yaml.safe_load(handle) or {})
