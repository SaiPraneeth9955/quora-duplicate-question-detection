import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qqp.inference import Prediction

DEMO = Path(__file__).resolve().parents[1] / "app" / "demo.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDetector:
    def __init__(self, name, threshold, score):
        self.name, self.threshold, self._score = name, threshold, score

    def predict(self, question1, question2):
        return Prediction(self.name, self._score, self.threshold)


@pytest.fixture
def client(demo):
    detectors = {
        "tfidf": FakeDetector("TF-IDF + cosine", 0.2, 0.1),
        "sbert_finetuned": FakeDetector("Fine-tuned Sentence-BERT", 0.8, 0.9),
    }
    validation = {"sbert_finetuned": {"f1": 0.856, "roc_auc": 0.953}}
    return TestClient(demo.create_app(detectors, "sbert_finetuned", validation))


def test_models_endpoint_lists_models_and_best(client):
    payload = client.get("/api/models").json()
    assert payload["best"] == "sbert_finetuned"
    assert [m["slug"] for m in payload["models"]] == ["tfidf", "sbert_finetuned"]
    assert payload["models"][1]["validation_f1"] == 0.856
    assert payload["models"][0]["validation_f1"] is None


def test_compare_returns_every_model_and_the_best_verdict(client):
    response = client.post("/api/compare", json={"question1": " A? ", "question2": "B?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["best"]["slug"] == "sbert_finetuned"
    assert payload["best"]["is_duplicate"] is True
    assert [r["label"] for r in payload["results"]] == ["Not duplicate", "Duplicate"]


@pytest.mark.parametrize("body", [{"question1": "  ", "question2": "B?"}, {"question1": "A?"}])
def test_compare_rejects_missing_questions(client, body):
    assert client.post("/api/compare", json=body).status_code == 422


def test_warm_endpoint_runs_without_a_body(client):
    assert client.post("/api/warm").status_code == 204


def test_page_and_assets_are_served(client):
    assert "Duplicate Question Detector" in client.get("/").text
    for asset in ("styles.css", "app.js", "favicon.svg"):
        assert client.get(f"/static/{asset}").status_code == 200
