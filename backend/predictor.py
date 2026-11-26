import os
import json
import joblib
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.preprocess import clean_transaction_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


def artifacts_exist() -> bool:
    return os.path.exists(os.path.join(ARTIFACTS_DIR, "baseline_model.pkl"))


def load_models():
    baseline_bundle = joblib.load(os.path.join(ARTIFACTS_DIR, "baseline_model.pkl"))
    bert_dir = os.path.join(ARTIFACTS_DIR, "distilbert_txn_model")
    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
    bert_model.eval()

    with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    return baseline_bundle, tokenizer, bert_model, metrics


def predict_transaction(raw_text: str, baseline_bundle, tokenizer, bert_model):
    cleaned = clean_transaction_text(raw_text)
    labels = [baseline_bundle["id2label"][i] for i in range(len(baseline_bundle["id2label"]))]

    # 1. Classical Baseline
    vec = baseline_bundle["vectorizer"].transform([cleaned])
    base_probs = baseline_bundle["model"].predict_proba(vec)[0]
    base_idx = int(base_probs.argmax())
    base_label = baseline_bundle["id2label"][base_idx]
    base_conf = float(base_probs[base_idx])

    # 2. Fine-tuned DistilBERT
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    bert_idx = int(probs.argmax())
    bert_label = bert_model.config.id2label[bert_idx]
    bert_conf = float(probs[bert_idx])

    prob_df = pd.DataFrame({
        "Category": labels * 2,
        "Confidence (%)": list(base_probs * 100) + list(probs * 100),
        "Model": ["TF-IDF + LogReg"] * len(labels) + ["DistilBERT"] * len(labels)
    })

    return cleaned, (base_label, base_conf), (bert_label, bert_conf), prob_df