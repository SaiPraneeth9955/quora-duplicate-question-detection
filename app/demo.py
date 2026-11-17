"""Interactive demo: enter two questions and see every trained model's verdict.

python app/demo.py
"""

import argparse
from functools import cache

import gradio as gr
import pandas as pd

from qqp.inference import available_models, best_model_slug, load_detector

EXAMPLES = [
    ["How can I learn Python?", "What is the best way to study Python?"],
    ["How can I learn Python?", "What is Java programming?"],
    ["How do I lose weight quickly?", "What is the fastest way to lose weight?"],
    ["Why is the sky blue?", "Why is the ocean blue?"],
    ["How do I become a data scientist?", "What does a data scientist do?"],
]


def build_app(artifacts_dir: str, results_dir: str) -> gr.Blocks:
    slugs = available_models(artifacts_dir)
    if not slugs:
        raise SystemExit(f"No trained models in '{artifacts_dir}'. Run `qqp train all` first.")
    try:
        best = best_model_slug(results_dir)
    except FileNotFoundError:
        best = slugs[-1]

    @cache
    def detector(slug: str):
        return load_detector(slug, artifacts_dir, results_dir)

    names = {slug: detector(slug).name for slug in slugs}

    def classify(question1: str, question2: str):
        if not question1.strip() or not question2.strip():
            raise gr.Error("Enter both questions.")
        predictions = {slug: detector(slug).predict(question1, question2) for slug in slugs}
        headline = predictions[best]
        verdict = (
            f"## {headline.label}\n"
            f"{names[best]} scored this pair **{headline.score:.3f}** "
            f"against a decision threshold of {headline.threshold:.3f}."
        )
        table = pd.DataFrame(
            {
                "Model": [names[s] + (" (best)" if s == best else "") for s in slugs],
                "Score": [f"{predictions[s].score:.3f}" for s in slugs],
                "Threshold": [f"{predictions[s].threshold:.3f}" for s in slugs],
                "Prediction": [predictions[s].label for s in slugs],
            }
        )
        return verdict, table

    with gr.Blocks(title="Duplicate Question Detector") as app:
        gr.Markdown(
            "# Duplicate Question Detector\n"
            "Do two questions ask the same thing? Models trained on Quora Question Pairs."
        )
        with gr.Row():
            question1 = gr.Textbox(label="Question 1", lines=2)
            question2 = gr.Textbox(label="Question 2", lines=2)
        submit = gr.Button("Compare", variant="primary")
        verdict = gr.Markdown()
        table = gr.Dataframe(label="All models", interactive=False)
        gr.Examples(EXAMPLES, inputs=[question1, question2])

        submit.click(classify, [question1, question2], [verdict, table])
        question2.submit(classify, [question1, question2], [verdict, table])
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(args.artifacts_dir, args.results_dir).launch(server_port=args.port)
