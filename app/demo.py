"""Web demo: enter two questions and see every trained model's verdict.

    python app/demo.py                 # serves http://127.0.0.1:7860 and opens the browser
    python app/demo.py --no-browser    # serve only

The page and API run entirely on this computer; nothing is sent elsewhere.
"""

import argparse
import logging
import os
import threading
import time
import webbrowser
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _prefer_local_models(artifacts_dir: Path) -> None:
    """Skip the Hugging Face version check when the pretrained encoder is already cached.

    The check is the only network call the demo makes. Skipping it makes start-up
    fast on poor connections and guarantees the evaluated weights are the ones used.
    Must run before transformers or huggingface_hub are imported.
    """
    import json

    meta = artifacts_dir / "sbert" / "meta.json"
    if not meta.exists():
        return
    model_name = json.loads(meta.read_text(encoding="utf-8"))["config"]["model_name"]
    cached = _hf_cache_dir() / ("models--" + model_name.replace("/", "--"))
    if cached.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _quiet_libraries() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda record: record.levelno >= logging.ERROR)


WARMUP_PAIRS = [
    ("Why?", "How?"),
    ("How can I learn Python?", "What is the best way to study Python?"),
    (
        "What are the most effective strategies to prepare for a machine learning interview?",
        "How should I study for data science and deep learning job interviews this year?",
    ),
]


def create_app(
    detectors: dict,
    best: str,
    validation: dict[str, dict] | None = None,
    warmup: bool = False,
):
    """Build the web app around already-loaded detectors.

    ``detectors`` maps model slugs to objects with ``name``, ``threshold`` and
    ``predict(question1, question2)``; ``validation`` maps slugs to their mean
    validation metrics, shown alongside each model.

    All scoring runs on one dedicated thread. GPU libraries set up per-thread state
    on first use, so with ``warmup`` that cost is paid at start-up, on the same
    thread that later serves every request, instead of on the first live request.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    validation = validation or {}
    scorer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scorer")
    if warmup:
        warm = [(d, pair) for pair in WARMUP_PAIRS for d in detectors.values()]
        scorer.submit(lambda: [d.predict(*pair) for d, pair in warm]).result()

    class Pair(BaseModel):
        question1: str = Field(max_length=1000)
        question2: str = Field(max_length=1000)

    def describe(slug: str) -> dict:
        detector = detectors[slug]
        metrics = validation.get(slug, {})
        return {
            "slug": slug,
            "name": detector.name,
            "threshold": detector.threshold,
            "is_best": slug == best,
            "validation_f1": metrics.get("f1"),
            "validation_roc_auc": metrics.get("roc_auc"),
        }

    app = FastAPI(title="Duplicate Question Detector", docs_url=None, redoc_url=None)

    @app.get("/api/models")
    def models() -> dict:
        return {"best": best, "models": [describe(slug) for slug in detectors]}

    def score_all(question1: str, question2: str) -> list[dict]:
        results = []
        for slug, detector in detectors.items():
            prediction = detector.predict(question1, question2)
            results.append(
                {
                    **describe(slug),
                    "score": prediction.score,
                    "is_duplicate": prediction.is_duplicate,
                    "label": prediction.label,
                }
            )
        return results

    def keep_gpu_awake() -> None:
        import torch

        if torch.cuda.is_available():
            torch.ones(1, device="cuda").sum().item()

    @app.post("/api/warm", status_code=204)
    async def warm() -> None:
        # A laptop GPU drops into power saving after ~10 s idle and then needs well
        # over a second to wake. The open page calls this every few seconds while
        # visible, so a comparison never pays that cost; a closed tab lets it sleep.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(scorer, keep_gpu_awake)

    @app.post("/api/compare")
    async def compare(pair: Pair) -> dict:
        question1, question2 = pair.question1.strip(), pair.question2.strip()
        if not question1 or not question2:
            raise HTTPException(status_code=422, detail="Enter both questions.")
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(scorer, score_all, question1, question2)
        return {
            "best": next(r for r in results if r["slug"] == best),
            "results": results,
            "elapsed_ms": round(1000 * (time.perf_counter() - started)),
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def load_app(artifacts_dir: str, results_dir: str, device: str | None = None):
    from qqp.inference import available_models, best_model_slug, load_detector
    from qqp.utils import read_json

    slugs = available_models(artifacts_dir)
    if not slugs:
        raise SystemExit(f"No trained models in '{artifacts_dir}'. Run `qqp train all` first.")
    try:
        best = best_model_slug(results_dir)
    except FileNotFoundError:
        best = slugs[-1]

    summary_path = Path(results_dir) / "summary.json"
    validation = {}
    if summary_path.exists():
        for slug, metrics in read_json(summary_path).get("models", {}).items():
            validation[slug] = {
                key: value["mean"] for key, value in metrics.items() if isinstance(value, dict)
            }

    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    detectors = {
        slug: load_detector(slug, artifacts_dir, results_dir, device=device) for slug in slugs
    }
    return create_app(detectors, best, validation, warmup=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Duplicate question detection demo")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()

    _prefer_local_models(Path(args.artifacts_dir))
    _quiet_libraries()
    print("Loading models...", flush=True)
    app = load_app(args.artifacts_dir, args.results_dir, args.device)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Duplicate Question Detector is ready at {url}  (press Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
