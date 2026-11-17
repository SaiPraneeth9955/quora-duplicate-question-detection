import numpy as np
import pandas as pd
import pytest
import torch

from qqp.config import RnnConfig
from qqp.models.sbert import online_contrastive_loss
from qqp.models.siamese import SiameseClassifier, SiameseLSTM, Vocabulary
from qqp.models.tfidf import TfidfSimilarity

PAIRS = pd.DataFrame(
    {
        "question1": ["How can I learn Python?", "What is the capital of France?"] * 8,
        "question2": ["What is the best way to learn Python?", "How do I bake bread?"] * 8,
        "label": [1, 0] * 8,
    }
)


def test_vocabulary_encoding_pads_truncates_and_handles_empty_text():
    vocab = Vocabulary.build(["a b b c", "b c"], max_size=10, min_freq=2)
    assert vocab.itos == ["<pad>", "<unk>", "b", "c"]
    ids, lengths = vocab.encode(["b c zzz", "", "b b b b b b"], max_len=4)
    assert ids.tolist() == [[2, 3, 1, 0], [1, 0, 0, 0], [2, 2, 2, 2]]
    assert lengths.tolist() == [3, 1, 4]


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("pooling", ["max", "last"])
def test_siamese_network_is_symmetric_and_ignores_padding(bidirectional, pooling):
    torch.manual_seed(0)
    network = SiameseLSTM(20, 8, 6, 0.0, bidirectional=bidirectional, pooling=pooling).eval()
    a = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 0, 0, 0]])
    b = torch.tensor([[8, 9, 0, 0, 0], [3, 4, 5, 6, 7]])
    len_a, len_b = torch.tensor([3, 2]), torch.tensor([2, 5])

    forward = network(a, len_a, b, len_b)
    assert forward.shape == (2,)
    assert torch.allclose(forward, network(b, len_b, a, len_a), atol=1e-6)

    # Extra padding columns must not change the encoding.
    padded = torch.cat([a, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    assert torch.allclose(network.encode(a, len_a), network.encode(padded, len_a), atol=1e-6)


def test_siamese_classifier_round_trips_through_disk(tmp_path):
    cfg = RnnConfig(max_len=8, embedding_dim=8, hidden_dim=8, batch_size=4, epochs=2, min_freq=1)
    model = SiameseClassifier(cfg, bidirectional=True, device="cpu")
    history = model.fit(PAIRS, PAIRS, seed=0)
    assert len(history) == 2

    before = model.score(PAIRS["question1"], PAIRS["question2"])
    model.save(tmp_path)
    after = SiameseClassifier.load(tmp_path, device="cpu").score(
        PAIRS["question1"], PAIRS["question2"]
    )
    assert before.shape == (len(PAIRS),)
    assert np.all((before >= 0) & (before <= 1))
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_tfidf_scores_and_round_trip(tmp_path):
    model = TfidfSimilarity().fit(pd.concat([PAIRS["question1"], PAIRS["question2"]]))
    scores = model.score(
        ["How can I learn Python?", "How can I learn Python?", "zzz qqq"],
        ["how can i learn python", "How do I bake bread?", "How do I bake bread?"],
    )
    assert scores[0] == pytest.approx(1.0)
    assert 0 <= scores[1] < scores[0]
    assert scores[2] == 0.0  # out-of-vocabulary question

    model.save(tmp_path)
    reloaded = TfidfSimilarity.load(tmp_path)
    np.testing.assert_allclose(
        reloaded.score(PAIRS["question1"], PAIRS["question2"]),
        model.score(PAIRS["question1"], PAIRS["question2"]),
    )


def test_online_contrastive_loss_only_counts_hard_pairs():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])

    # Duplicates closer than every non-duplicate: nothing is hard.
    easy = torch.tensor([0.1, 0.2, 0.8, 0.9])
    assert online_contrastive_loss(easy, labels, margin=0.5).item() == 0.0

    # One duplicate at 0.6 is further than the non-duplicate at 0.3.
    hard = torch.tensor([0.1, 0.6, 0.3, 0.9])
    expected = 0.6**2 + (0.5 - 0.3) ** 2
    assert online_contrastive_loss(hard, labels, margin=0.5).item() == pytest.approx(expected)

    # A batch with a single class falls back to plain contrastive loss.
    positives_only = online_contrastive_loss(torch.tensor([0.2, 0.4]), torch.ones(2), margin=0.5)
    assert positives_only.item() == pytest.approx(0.2**2 + 0.4**2)
