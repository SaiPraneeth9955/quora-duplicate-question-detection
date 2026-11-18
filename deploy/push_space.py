"""Publish the web demo to a Hugging Face Space.

    hf auth login                                     # once, with a token that can write
    python deploy/push_space.py duplicate-question-detector

Uploads the container recipe in deploy/huggingface/, the qqp package, the web app,
the trained models from artifacts/ and results/summary.json. Hugging Face then
builds the container and serves the app at https://<user>-<space>.hf.space.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("tfidf", "siamese_lstm", "siamese_bilstm", "sbert", "sbert_finetuned")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def stage(target: Path) -> None:
    """Assemble exactly the files the Space needs."""
    shutil.copytree(ROOT / "deploy" / "huggingface", target, dirs_exist_ok=True)
    shutil.copytree(ROOT / "src" / "qqp", target / "src" / "qqp", ignore=IGNORE)
    shutil.copytree(ROOT / "app", target / "app", ignore=IGNORE)
    for slug in MODELS:
        source = ROOT / "artifacts" / slug
        if not (source / "meta.json").exists():
            raise SystemExit(f"Missing trained model in {source}. Run `qqp train {slug}` first.")
        shutil.copytree(source, target / "artifacts" / slug, ignore=IGNORE)
    (target / "results").mkdir()
    shutil.copy2(ROOT / "results" / "summary.json", target / "results" / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("space", help="Space name, or <user>/<name>")
    parser.add_argument("--private", action="store_true", help="create the Space as private")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception as error:
        raise SystemExit(f"Not signed in to Hugging Face ({error}). Run `hf auth login`.") from None
    repo_id = args.space if "/" in args.space else f"{user}/{args.space}"

    api.create_repo(
        repo_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True
    )
    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp))
        size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file()) / 1e6
        print(f"Uploading {size:.0f} MB to {repo_id} ...", flush=True)
        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy duplicate question detector",
            # Remove files from earlier uploads that no longer exist locally.
            delete_patterns=["src/**", "app/**", "artifacts/**", "results/**"],
        )

    owner, name = repo_id.split("/")
    print(f"Space:    https://huggingface.co/spaces/{repo_id}")
    print(f"Live app: https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space")
    print("Hugging Face is now building the container; the app is live in about 5-10 minutes.")


if __name__ == "__main__":
    main()
