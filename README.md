# Phishing Email Payload Detection using DistilBERT

This repository contains a clean, GitHub-ready Python implementation for phishing email/payload detection.

## Main file

`phishing_email_detection_distilbert_correct.py`

## What it does

- Loads the public Hugging Face dataset `zefang-liu/phishing-email-dataset`
- Cleans email text and maps labels:
  - `Safe Email` -> 0
  - `Phishing Email` -> 1
- Splits the original dataset before augmentation to reduce leakage risk
- Applies adversarial-style augmentation only to the training split
- Supports two approaches:
  1. Fine-tuned DistilBERT sequence classification
  2. DistilBERT CLS embeddings + classical ML classifiers
- Saves metrics, classification reports, confusion matrices, ROC curves, and model comparison CSV files

## Quick usage

Print documented original notebook results only:

```bash
python phishing_email_detection_distilbert_correct.py --mode reported
```

Run DistilBERT embeddings + classical ML benchmark:

```bash
python phishing_email_detection_distilbert_correct.py --mode embeddings
```

Fine-tune DistilBERT:

```bash
python phishing_email_detection_distilbert_correct.py --mode finetune --epochs 2
```

## Reported original notebook results

Fine-tuned DistilBERT achieved:

| Metric | Score |
|---|---:|
| Accuracy | 98.52% |
| Precision | 96.84% |
| Recall | 99.49% |
| F1-score | 98.14% |
| ROC-AUC | 0.9993 |

