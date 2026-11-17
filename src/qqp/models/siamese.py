"""Siamese LSTM / BiLSTM duplicate classifier."""

import logging
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from qqp.config import RnnConfig
from qqp.data import normalize_text
from qqp.utils import read_json, resolve_device, set_seed, write_json

log = logging.getLogger(__name__)


class Vocabulary:
    PAD, UNK = 0, 1

    def __init__(self, tokens: Sequence[str]):
        self.itos = ["<pad>", "<unk>", *tokens]
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int, min_freq: int) -> "Vocabulary":
        counts = Counter(token for text in texts for token in text.split())
        return cls([t for t, c in counts.most_common(max_size) if c >= min_freq])

    def encode(self, texts: Sequence[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Map normalised texts to a zero-padded id matrix and their lengths.

        An empty question is encoded as a single ``<unk>`` so that every
        sequence has length >= 1, which packed sequences require.
        """
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        lengths = np.ones(len(texts), dtype=np.int64)
        for row, text in enumerate(texts):
            tokens = [self.stoi.get(t, self.UNK) for t in text.split()[:max_len]] or [self.UNK]
            ids[row, : len(tokens)] = tokens
            lengths[row] = len(tokens)
        return torch.from_numpy(ids), torch.from_numpy(lengths)


class SiameseLSTM(nn.Module):
    """One encoder applied to both questions, followed by an MLP on their interaction.

    A question is encoded either as the element-wise maximum over the LSTM's
    hidden states (``pooling="max"``) or as its final hidden state
    (``pooling="last"``). The pair is represented as ``[|u - v|, u * v]``,
    which is symmetric in the two questions: swapping them cannot change the
    prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        bidirectional: bool,
        pooling: str = "max",
    ):
        super().__init__()
        if pooling not in ("max", "last"):
            raise ValueError(f"pooling must be 'max' or 'last', got '{pooling}'")
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=Vocabulary.PAD)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional
        )
        encoding_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.dropout(self.embedding(ids)), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, _) = self.lstm(packed)
        if self.pooling == "max":
            states, _ = pad_packed_sequence(outputs, batch_first=True, total_length=ids.size(1))
            padding = (ids == Vocabulary.PAD).unsqueeze(-1)
            return states.masked_fill(padding, float("-inf")).max(dim=1).values
        if self.bidirectional:
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        return hidden[-1]

    def forward(self, ids1, lengths1, ids2, lengths2) -> torch.Tensor:
        u = self.encode(ids1, lengths1)
        v = self.encode(ids2, lengths2)
        return self.classifier(torch.cat([(u - v).abs(), u * v], dim=1)).squeeze(1)


class SiameseClassifier:
    """Training and inference wrapper around :class:`SiameseLSTM`."""

    def __init__(
        self,
        cfg: RnnConfig | None = None,
        bidirectional: bool = False,
        device: str | None = None,
    ):
        self.cfg = cfg or RnnConfig()
        self.bidirectional = bidirectional
        self.device = resolve_device(device)
        self.vocab: Vocabulary | None = None
        self.network: SiameseLSTM | None = None

    def _build_network(self) -> SiameseLSTM:
        return SiameseLSTM(
            len(self.vocab),
            self.cfg.embedding_dim,
            self.cfg.hidden_dim,
            self.cfg.dropout,
            self.bidirectional,
            self.cfg.pooling,
        ).to(self.device)

    def _encode_pairs(self, questions1: Sequence[str], questions2: Sequence[str]):
        ids1, len1 = self.vocab.encode([normalize_text(q) for q in questions1], self.cfg.max_len)
        ids2, len2 = self.vocab.encode([normalize_text(q) for q in questions2], self.cfg.max_len)
        return ids1, len1, ids2, len2

    @torch.no_grad()
    def _predict(self, encoded, labels: torch.Tensor | None = None, batch_size: int = 2048):
        """Return duplicate probabilities and, if labels are given, the mean loss."""
        self.network.eval()
        ids1, len1, ids2, len2 = encoded
        probabilities, loss_sum = [], 0.0
        for start in range(0, len(ids1), batch_size):
            batch = slice(start, start + batch_size)
            logits = self.network(
                ids1[batch].to(self.device), len1[batch], ids2[batch].to(self.device), len2[batch]
            )
            if labels is not None:
                loss_sum += F.binary_cross_entropy_with_logits(
                    logits, labels[batch].to(self.device), reduction="sum"
                ).item()
            probabilities.append(torch.sigmoid(logits).cpu())
        probabilities = torch.cat(probabilities).numpy()
        return probabilities, (loss_sum / len(ids1) if labels is not None else None)

    def fit(self, train: pd.DataFrame, tune: pd.DataFrame, seed: int = 42) -> list[dict]:
        """Train with early stopping on tuning loss; returns the per-epoch history."""
        cfg = self.cfg
        set_seed(seed)

        normalised = (normalize_text(q) for col in ("question1", "question2") for q in train[col])
        self.vocab = Vocabulary.build(normalised, cfg.max_vocab, cfg.min_freq)
        self.network = self._build_network()
        log.info(
            "vocabulary %d tokens, %.1fM parameters",
            len(self.vocab),
            sum(p.numel() for p in self.network.parameters()) / 1e6,
        )

        ids1, len1, ids2, len2 = self._encode_pairs(train["question1"], train["question2"])
        labels = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
        tune_encoded = self._encode_pairs(tune["question1"], tune["question2"])
        tune_labels = torch.tensor(tune["label"].to_numpy(), dtype=torch.float32)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.lr)
        # Halve the learning rate whenever an epoch fails to improve the tuning loss.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.lr_decay, patience=0
        )
        history, best_loss, best_state, stale_epochs = [], float("inf"), None, 0
        n = len(labels)

        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            self.network.train()
            order = torch.randperm(n)
            loss_sum, correct = 0.0, 0
            for start in range(0, n, cfg.batch_size):
                index = order[start : start + cfg.batch_size]
                target = labels[index].to(self.device)
                logits = self.network(
                    ids1[index].to(self.device),
                    len1[index],
                    ids2[index].to(self.device),
                    len2[index],
                )
                loss = F.binary_cross_entropy_with_logits(logits, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                optimizer.step()
                loss_sum += loss.item() * len(index)
                correct += ((logits > 0) == (target > 0.5)).sum().item()

            tune_probabilities, tune_loss = self._predict(tune_encoded, tune_labels)
            tune_accuracy = float(
                ((tune_probabilities > 0.5) == (tune_labels.numpy() > 0.5)).mean()
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": loss_sum / n,
                    "train_accuracy": correct / n,
                    "tune_loss": tune_loss,
                    "tune_accuracy": tune_accuracy,
                    "seconds": time.perf_counter() - started,
                }
            )
            log.info(
                "epoch %2d  train loss %.4f acc %.4f | tune loss %.4f acc %.4f  (%.0fs)",
                epoch,
                history[-1]["train_loss"],
                history[-1]["train_accuracy"],
                tune_loss,
                tune_accuracy,
                history[-1]["seconds"],
            )

            scheduler.step(tune_loss)
            if tune_loss < best_loss:
                best_loss, stale_epochs = tune_loss, 0
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    log.info(
                        "early stopping: tuning loss has not improved for %d epochs", cfg.patience
                    )
                    break

        self.network.load_state_dict(best_state)
        return history

    def score(self, questions1: Sequence[str], questions2: Sequence[str]) -> np.ndarray:
        probabilities, _ = self._predict(self._encode_pairs(questions1, questions2))
        return probabilities

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / "weights.pt")
        write_json(directory / "vocab.json", self.vocab.itos[2:])
        write_json(
            directory / "architecture.json",
            {
                "bidirectional": self.bidirectional,
                "max_len": self.cfg.max_len,
                "embedding_dim": self.cfg.embedding_dim,
                "hidden_dim": self.cfg.hidden_dim,
                "dropout": self.cfg.dropout,
                "pooling": self.cfg.pooling,
            },
        )

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> "SiameseClassifier":
        directory = Path(directory)
        architecture = read_json(directory / "architecture.json")
        bidirectional = architecture.pop("bidirectional")
        model = cls(RnnConfig(**architecture), bidirectional=bidirectional, device=device)
        model.vocab = Vocabulary(read_json(directory / "vocab.json"))
        model.network = model._build_network()
        state = torch.load(directory / "weights.pt", map_location=model.device, weights_only=True)
        model.network.load_state_dict(state)
        model.network.eval()
        return model
