"""Sentence-BERT cosine similarity, used as-is or fine-tuned on QQP."""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup

from qqp.config import SbertConfig
from qqp.metrics import best_f1_threshold, classification_metrics
from qqp.utils import resolve_device, set_seed

log = logging.getLogger(__name__)


def online_contrastive_loss(
    distances: torch.Tensor, labels: torch.Tensor, margin: float
) -> torch.Tensor:
    """Contrastive loss restricted to the hard pairs in the batch.

    A duplicate pair is hard when it is further apart than the closest
    non-duplicate; a non-duplicate pair is hard when it is closer than the
    furthest duplicate. Easy pairs contribute nothing, which keeps the
    gradient focused on the pairs near the decision boundary.
    """
    positives = distances[labels == 1]
    negatives = distances[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        hard_positives, hard_negatives = positives, negatives
    else:
        hard_positives = positives[positives > negatives.min()]
        hard_negatives = negatives[negatives < positives.max()]
    return hard_positives.pow(2).sum() + F.relu(margin - hard_negatives).pow(2).sum()


class SbertSimilarity:
    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        cfg: SbertConfig | None = None,
        device: str | None = None,
    ):
        self.cfg = cfg or SbertConfig()
        self.device = resolve_device(device)
        self.model = SentenceTransformer(
            str(model_name_or_path or self.cfg.model_name), device=self.device
        )
        self.model.max_seq_length = self.cfg.max_seq_length

    def score(self, questions1: Sequence[str], questions2: Sequence[str]) -> np.ndarray:
        options = dict(
            batch_size=self.cfg.eval_batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        a = self.model.encode(list(questions1), **options)
        b = self.model.encode(list(questions2), **options)
        return (a * b).sum(dim=1).float().cpu().numpy()

    def _embed(self, texts: list[str]) -> torch.Tensor:
        features = self.model.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_seq_length,
            return_tensors="pt",
        )
        features = {name: tensor.to(self.device) for name, tensor in features.items()}
        return F.normalize(self.model(features)["sentence_embedding"].float(), dim=1)

    def _evaluate(self, frame: pd.DataFrame) -> dict:
        labels = frame["label"].to_numpy()
        scores = self.score(frame["question1"], frame["question2"])
        threshold = best_f1_threshold(labels, scores)
        return {
            "tune_roc_auc": float(roc_auc_score(labels, scores)),
            "tune_f1": classification_metrics(labels, scores, threshold)["f1"],
        }

    def finetune(self, train: pd.DataFrame, tune: pd.DataFrame, seed: int = 42) -> list[dict]:
        """Fine-tune on labelled pairs; returns the training history.

        Each history entry holds the mean training loss since the previous
        entry and the tuning-split ROC-AUC / F1 at that step.
        """
        cfg = self.cfg
        set_seed(seed)

        questions1 = train["question1"].tolist()
        questions2 = train["question2"].tolist()
        labels = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)

        n = len(train)
        steps_per_epoch = (n + cfg.batch_size - 1) // cfg.batch_size
        total_steps = cfg.epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(cfg.warmup_fraction * total_steps), total_steps
        )
        use_amp = self.device.startswith("cuda")
        scaler = torch.amp.GradScaler(enabled=use_amp)
        device_type = "cuda" if use_amp else "cpu"

        history = [{"step": 0, "train_loss": None, **self._evaluate(tune)}]
        log.info(
            "step 0/%d  tune AUC %.4f F1 %.4f (pretrained)",
            total_steps,
            history[0]["tune_roc_auc"],
            history[0]["tune_f1"],
        )

        step, running_loss, running_steps = 0, 0.0, 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n).tolist()
            self.model.train()
            for start in range(0, n, cfg.batch_size):
                index = order[start : start + cfg.batch_size]
                target = labels[index].to(self.device)
                with torch.autocast(device_type=device_type, enabled=use_amp):
                    u = self._embed([questions1[i] for i in index])
                    v = self._embed([questions2[i] for i in index])
                distances = 1 - (u * v).sum(dim=1)
                loss = online_contrastive_loss(distances, target, cfg.margin)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                step += 1
                running_loss += loss.item() / len(index)
                running_steps += 1
                if step % cfg.eval_every == 0 or step == total_steps:
                    entry = {
                        "step": step,
                        "train_loss": running_loss / running_steps,
                        **self._evaluate(tune),
                    }
                    history.append(entry)
                    log.info(
                        "step %d/%d  train loss %.5f | tune AUC %.4f F1 %.4f",
                        step,
                        total_steps,
                        entry["train_loss"],
                        entry["tune_roc_auc"],
                        entry["tune_f1"],
                    )
                    running_loss, running_steps = 0.0, 0
                    self.model.train()

        self.model.eval()
        return history

    def save(self, directory: str | Path) -> None:
        self.model.save(str(directory))

    @classmethod
    def load(
        cls, directory: str | Path, cfg: SbertConfig | None = None, device: str | None = None
    ) -> "SbertSimilarity":
        return cls(directory, cfg=cfg, device=device)
