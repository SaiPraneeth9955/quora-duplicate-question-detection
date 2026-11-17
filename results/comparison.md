| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Threshold | Train time (min) | Inference (pairs/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + cosine | 0.5823 | 0.4655 | 0.9090 | 0.6157 | 0.6879 | 0.156 | 0.2 | 29,161 |
| Siamese LSTM | 0.8195 | 0.7308 | 0.8073 | 0.7671 | 0.9011 | 0.634 | 5.6 | 55,921 |
| Siamese BiLSTM | 0.8183 | 0.7146 | 0.8429 | 0.7735 | 0.9048 | 0.588 | 5.7 | 45,881 |
| Sentence-BERT + cosine | 0.7638 | 0.6250 | 0.8961 | 0.7364 | 0.8707 | 0.734 | - | 2,282 |
| Fine-tuned Sentence-BERT | 0.8871 | 0.8039 | 0.9168 | 0.8567 | 0.9541 | 0.799 | 26.5 | 2,139 |
