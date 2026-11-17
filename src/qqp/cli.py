"""Command-line interface: ``qqp train``, ``qqp report``, ``qqp predict``."""

import argparse
import logging
import os
from pathlib import Path

from qqp.config import Config, load_config

DEFAULT_CONFIG = Path("configs/default.yaml")


def _config_from_args(args: argparse.Namespace) -> Config:
    path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cfg = load_config(path)
    if getattr(args, "subsample", None) is not None:
        cfg.data.subsample = args.subsample
    if getattr(args, "seeds", None):
        cfg.seeds = args.seeds
    if args.artifacts_dir:
        cfg.artifacts_dir = args.artifacts_dir
    if args.results_dir:
        cfg.results_dir = args.results_dir
    return cfg


def _train(args: argparse.Namespace) -> None:
    from qqp.data import load_splits
    from qqp.pipeline import MODELS, run_model
    from qqp.report import build_report

    cfg = _config_from_args(args)
    slugs = list(MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    splits = load_splits(cfg.data, cfg.split_seed)
    for slug in slugs:
        run_model(slug, cfg, splits, device=args.device, force=args.force)
    print(build_report(cfg.results_dir))


def _report(args: argparse.Namespace) -> None:
    from qqp.report import build_report

    cfg = _config_from_args(args)
    print(build_report(cfg.results_dir))


def _predict(args: argparse.Namespace) -> None:
    from qqp.inference import load_detector

    cfg = _config_from_args(args)
    detector = load_detector(args.model, cfg.artifacts_dir, cfg.results_dir, device=args.device)
    prediction = detector.predict(args.question1, args.question2)
    print(
        f"{prediction.label}  "
        f"(score {prediction.score:.3f}, threshold {prediction.threshold:.3f}, "
        f"model: {prediction.model})"
    )


def build_parser() -> argparse.ArgumentParser:
    model_choices = ["tfidf", "siamese_lstm", "siamese_bilstm", "sbert", "sbert_finetuned"]

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, help=f"YAML config (default: {DEFAULT_CONFIG})")
    common.add_argument("--artifacts-dir", help="where trained models are stored")
    common.add_argument("--results-dir", help="where metrics and figures are stored")
    common.add_argument("--device", help="cuda or cpu (default: cuda when available)")

    parser = argparse.ArgumentParser(prog="qqp", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", parents=[common], help="train and evaluate models")
    train.add_argument("models", nargs="+", choices=[*model_choices, "all"])
    train.add_argument("--subsample", type=int, help="use only N training pairs (quick runs)")
    train.add_argument("--seeds", type=int, nargs="+", help="training seeds (overrides config)")
    train.add_argument("--force", action="store_true", help="retrain runs that already exist")
    train.set_defaults(handler=_train)

    report = commands.add_parser("report", parents=[common], help="rebuild table and figures")
    report.set_defaults(handler=_report)

    predict = commands.add_parser("predict", parents=[common], help="classify one question pair")
    predict.add_argument("question1")
    predict.add_argument("question2")
    predict.add_argument("--model", default="best", choices=[*model_choices, "best"])
    predict.set_defaults(handler=_predict)
    return parser


def _quiet_third_party(record: logging.LogRecord) -> bool:
    return record.name.startswith("qqp") or record.levelno >= logging.ERROR


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    # Third-party libraries reset their own logger levels on import, so filter at the handler.
    for handler in logging.getLogger().handlers:
        handler.addFilter(_quiet_third_party)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args.handler(args)


if __name__ == "__main__":
    main()
