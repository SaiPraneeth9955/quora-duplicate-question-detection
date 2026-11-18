---
title: Quora Question Pair Semantic Similarity
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
short_description: Do two questions ask the same thing? Five models compared.
pinned: false
---

# Quora Question Pair Semantic Similarity

Enter two questions and see whether they ask the same thing. Five models trained
on [GLUE QQP](https://huggingface.co/datasets/nyu-mll/glue) score every pair side
by side; the fine-tuned Sentence-BERT model gives the verdict.

| Model | F1 (validation) | ROC-AUC |
|---|---:|---:|
| TF-IDF + cosine | 0.616 | 0.688 |
| Siamese LSTM | 0.769 | 0.903 |
| Siamese BiLSTM | 0.772 | 0.904 |
| Sentence-BERT + cosine | 0.694 | 0.827 |
| **Fine-tuned Sentence-BERT** | **0.856** | **0.953** |

Metrics are on the 40,430-pair GLUE validation set; trained models are the mean
of five training seeds. Each model's decision threshold was tuned on a held-out
split of the training data.

Code, training pipeline and full results:
[github.com/SaiPraneeth9955/quora-duplicate-question-detection](https://github.com/SaiPraneeth9955/quora-duplicate-question-detection)
