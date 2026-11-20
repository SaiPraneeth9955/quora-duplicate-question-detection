# Quora Question Pair Semantic Similarity

Given two questions, decide whether they ask the same thing. This repository
implements five approaches, from a bag-of-words baseline to a fine-tuned
transformer, and compares them under a single evaluation protocol on the
[GLUE QQP](https://huggingface.co/datasets/nyu-mll/glue) dataset.

| Question 1 | Question 2 | Label |
|---|---|---|
| How can I learn Python? | What is the best way to study Python? | Duplicate |
| How can I learn Python? | What is Java programming? | Not duplicate |

## Results

Official GLUE QQP validation set, 40,430 pairs, 36.8% duplicates. Trained
neural models are reported as mean ± standard deviation over five training
seeds; the two deterministic models are run once.

| Model | Runs | Accuracy | Precision | Recall | F1-score | ROC-AUC | Train time (min) | Inference (pairs/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + cosine | 1 | 0.5823 | 0.4655 | 0.9090 | 0.6157 | 0.6879 | 0.2 | 32,234 |
| Siamese LSTM | 5 | 0.8164 ± 0.0075 | 0.7161 ± 0.0164 | 0.8321 ± 0.0224 | 0.7694 ± 0.0079 | 0.9026 ± 0.0062 | 5.4 ± 1.4 | 57,694 |
| Siamese BiLSTM | 5 | 0.8179 ± 0.0049 | 0.7164 ± 0.0088 | 0.8369 ± 0.0072 | 0.7720 ± 0.0048 | 0.9041 ± 0.0034 | 5.5 ± 0.8 | 46,552 |
| Sentence-BERT + cosine | 1 | 0.7222 | 0.5837 | 0.8560 | 0.6941 | 0.8268 | - | 699 |
| **Fine-tuned Sentence-BERT** | 5 | **0.8881 ± 0.0012** | **0.8141 ± 0.0051** | **0.9023 ± 0.0050** | **0.8559 ± 0.0009** | **0.9531 ± 0.0002** | 68.3 ± 0.7 | 674 |

Training time and inference throughput were measured on an NVIDIA RTX 3050
Laptop GPU (4 GB); TF-IDF runs on the CPU. Every number above is produced by
`qqp train all` and stored under [`results/`](results); the metrics of each
individual run are in [`results/runs.csv`](results/runs.csv). The ROC curves
and confusion matrices show the seed-42 model, which is the one saved for
prediction.

![Validation metrics by model](results/figures/metrics.png)

| ROC curves | Confusion matrices |
|---|---|
| ![ROC curves](results/figures/roc_curves.png) | ![Confusion matrices](results/figures/confusion_matrices.png) |

Training curves:
[Siamese LSTM](results/figures/siamese_lstm_training.png) ·
[Siamese BiLSTM](results/figures/siamese_bilstm_training.png) ·
[Fine-tuned Sentence-BERT](results/figures/sbert_finetuned_training.png)

### Findings

- **Fine-tuned Sentence-BERT is the best model** on accuracy, precision, F1
  (0.856 ± 0.001) and ROC-AUC (0.953), and is the one selected for the demo.
  Only TF-IDF has higher recall (0.909 vs 0.902), and only because it labels
  most pairs as duplicates. Its
  worst seed (F1 0.8551) is still 7.85 points above the best LSTM or BiLSTM
  seed (0.7766), so the ranking does not depend on the seed.
- **Task-specific training matters more than pretraining.** The pretrained
  encoder used as-is (F1 0.694) is beaten by both LSTMs trained from scratch
  on QQP (F1 0.769 and 0.772). Fine-tuning the same encoder adds 16.2 points
  of F1 (0.6941 to 0.8559).
- **Bidirectionality makes no measurable difference.** The BiLSTM's mean F1 is
  0.0025 higher than the LSTM's, well inside the seed-to-seed spread (LSTM F1
  ranges from 0.7569 to 0.7766 across seeds). A Welch t-test over the five
  runs finds no significant difference in accuracy (p = 0.71), F1 (p = 0.56)
  or ROC-AUC (p = 0.66). The BiLSTM is, however, more consistent between seeds
  (F1 standard deviation 0.0048 vs 0.0079).
- **Word overlap is a weak signal.** TF-IDF only reaches 0.909 recall by
  flagging most pairs as duplicates; its accuracy (0.582) is below the 0.632
  obtained by always answering "not duplicate". Many non-duplicate pairs in
  QQP share most of their words.
- **Accuracy costs compute.** The LSTM and BiLSTM train in 5.4 and 5.5 minutes
  on average and score 57,694 and 46,552 pairs per second; the fine-tuned
  transformer takes 68.3 minutes to train and scores 674 pairs per second,
  85.6 and 69.1 times slower.

## Evaluation protocol

All five models are trained and scored on identical data.

| Split | Pairs | Source | Used for |
|---|---:|---|---|
| Train | 345,653 | 95% of GLUE train | Fitting model parameters |
| Tune | 18,193 | 5% of GLUE train, stratified | Early stopping, threshold selection |
| Validation | 40,430 | GLUE validation | Reported metrics only |

- Each model outputs one score per pair. Its decision threshold is the value
  that maximises F1 on the **tune** split, and is then applied unchanged to
  validation. Validation labels never influence training, model selection or
  thresholds.
- The TF-IDF vocabulary and the LSTM vocabulary are built from the train split
  only.
- F1 and ROC-AUC are the headline metrics because the classes are imbalanced
  (63.2% / 36.8%); accuracy alone rewards predicting "not duplicate".
- The partition is fixed (`split_seed: 42`), so every run of every model sees
  the same train, tune and validation pairs.
- **Repeated runs.** The three trained neural models depend on random
  initialisation and batch order, so each is trained five times with seeds
  42–46 and reported as mean ± sample standard deviation. TF-IDF and the
  pretrained encoder are deterministic and are run once.
- **Not trained on the benchmark.** The Sentence-BERT encoder,
  `nli-distilroberta-base-v2`, is DistilRoBERTa (pretrained on general web
  text, OpenWebTextCorpus) trained for sentence similarity on SNLI and
  MultiNLI; it was not trained on Quora Question Pairs. Widely used
  alternatives such as `all-MiniLM-L6-v2` list Quora duplicate questions among
  their training data, which could let validation pairs leak into the
  pretrained and fine-tuned rows.

## Models

| Model | Summary |
|---|---|
| **TF-IDF + cosine** | Unigram and bigram TF-IDF vectors; score is their cosine similarity. No learned notion of meaning, only word overlap. |
| **Siamese LSTM** | Shared embedding and LSTM encode each question; max-pooled states `u`, `v` are combined as `[\|u − v\|, u ⊙ v]` and classified by an MLP. Embeddings are trained from scratch. |
| **Siamese BiLSTM** | Same network with a bidirectional encoder. |
| **Sentence-BERT + cosine** | Pretrained `nli-distilroberta-base-v2` sentence embeddings, cosine similarity, no training. |
| **Fine-tuned Sentence-BERT** | The same encoder fine-tuned on the train split with an online contrastive loss, which updates only on the hard pairs in each batch. |

The Siamese pair representation is symmetric by construction, so swapping the
two questions cannot change a prediction.

## Getting started

Requires Python 3.10+ and, for the neural models, a CUDA GPU (4 GB is enough).

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[demo,dev]"
```

### Reproduce the experiment

```bash
qqp train all
```

This downloads the dataset, trains each model, and writes metrics, figures and
the comparison table to `results/` and trained models to `artifacts/`. The
17 runs spent 6 hours 42 minutes training and scoring the validation set on
the GPU above, 86% of it in the five fine-tuning runs (dataset download and
loading not included). Useful variations:

```bash
qqp train tfidf siamese_bilstm       # selected models only
qqp train all --subsample 20000 --seeds 42   # quick end-to-end check
qqp train all --force                # retrain runs that are already on disk
qqp report                           # rebuild table and figures from saved results
```

Each finished run is saved as it completes, so an interrupted `qqp train`
resumes from the first missing run when started again. The model trained with
the first seed is the one saved to `artifacts/` and used for prediction.

Hyperparameters live in [`configs/default.yaml`](configs/default.yaml); pass
`--config path.yaml` to use another file.

### Classify a pair

```bash
qqp predict "How can I learn Python?" "What is the best way to study Python?"
```

`--model` selects a specific model (`tfidf`, `siamese_lstm`, `siamese_bilstm`,
`sbert`, `sbert_finetuned`); the default is the model with the highest mean
validation F1.

From Python:

```python
from qqp.inference import load_detector

detector = load_detector("best")
prediction = detector.predict("How do I lose weight?", "What is the best way to lose weight?")
print(prediction.label, prediction.score)
```

### Demo app

```bash
python app/demo.py
```

Starts a local web app at <http://127.0.0.1:7860> and opens it in the browser.
Enter two questions to get the best model's verdict, its score against its
decision threshold, and every other model's score side by side. The app is a
small FastAPI service (`/api/compare`, `/api/models`) with a plain HTML page,
so it needs no internet connection; `--port` and `--no-browser` change the
defaults.

### Deploy online

The demo runs as a Docker container on [Hugging Face Spaces](https://huggingface.co/spaces)
on CPU-only hardware. With the GPU disabled on our laptop, one comparison with
all five models took 60 to 68 ms (mean 64.9 ms over 20 requests); the Space's
own speed has not been measured yet.

```bash
hf auth login                                        # once, with a write token
python deploy/push_space.py duplicate-question-detector
```

This uploads the container recipe in [`deploy/huggingface/`](deploy/huggingface),
the package, the web app and the trained models; Hugging Face builds the image
and serves the app at `https://<user>-duplicate-question-detector.hf.space`.

### Tests

```bash
pytest
```

## Repository layout

```
configs/default.yaml     hyperparameters for every model
src/qqp/
  data.py                dataset loading, text normalisation, splits
  metrics.py             threshold selection and evaluation metrics
  models/                tfidf.py, siamese.py, sbert.py
  pipeline.py            trains, evaluates and saves one model
  report.py              comparison table and figures
  inference.py           loads trained models for prediction
  cli.py                 the `qqp` command
app/demo.py              web demo: FastAPI service
app/static/              web demo: page, styles and script
tests/                   unit tests
results/                 comparison table, summary and figures (committed)
  runs.csv               metrics of every individual run
  <model>/<run>/         metrics and training history of one run
artifacts/               trained models (generated, not committed)
```

## Limitations

- Seeds vary model initialisation and batch order but not the train / tune
  partition, which is fixed so that every model sees identical data. The
  reported spread therefore excludes variation from resampling the data.
- The dataset authors state that the ground-truth labels "contain some amount
  of noise" ([Quora dataset release](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs)),
  which caps the accuracy any model can reach on this benchmark.
- The LSTMs learn embeddings from scratch; pretrained word vectors would
  likely narrow the gap to the transformer.
