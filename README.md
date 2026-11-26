# FinSense AI - Automated Bank & UPI Transaction Categorizer

**Course:** CSB306 - Natural Language Processing (Project ID: NLP-18)  
**Student:** S.syed basha (12405575)  
**Live Application:** https://finsense-ai-categorizer.streamlit.app

---

## Overview

Digital payment rails in India (UPI, NEFT, IMPS, POS, and ACH) generate noisy, unstructured transaction narrations such as:
* `Swiggy order payment | Ref:0f8f77bb | Amount: INR 450.00`
* `ACH/DEBIT/BAJAJ FINANCE EMI/99210`
* `UPI/CR/991827/ZERODHA BROKING LTD`

**FinSense AI** is a modular end-to-end NLP pipeline that strips banking-rail noise using regular expressions and classifies financial narrations into five personal-finance categories: **Food**, **Travel**, **EMI**, **Investment**, and **Shopping**.

---

## Project Architecture

```text
Bank-Transaction-Categorizer-NLP/
|-- backend/                          # Core NLP preprocessing, training & inference engine
|   |-- __init__.py                   # Package initializer for cross-directory imports
|   |-- preprocess.py                 # Regex noise scrubber (clean_transaction_text)
|   |-- train_models.py               # Reproducible training & evaluation benchmark script
|   |-- predictor.py                  # Inference API (load_models & predict_transaction)
|   +-- artifacts/                    # Serialized models, metrics & evaluation plots
|       |-- baseline_model.pkl        # TF-IDF (1,2-gram) + Logistic Regression pipeline
|       |-- distilbert_txn_model/     # Fine-tuned DistilBERT weights (tracked via Git LFS)
|       |-- confusion_matrix.png      # 1,000-sample test confusion matrix heatmap
|       |-- error_examples.csv        # Misclassified test samples log
|       +-- metrics.json              # Evaluated Accuracy & Macro F1 benchmarks
|-- frontend/                         # Interactive Streamlit SaaS dashboard & demo assets
|   |-- app.py                        # 3-tab UI (Live Tester, Batch Analyzer, Benchmarks)
|   +-- sample_statement.csv          # 8-row sample bank statement for one-click testing
|-- data/                             # Labeled Kaggle Indian Banking Transaction splits
|   |-- financial_transaction_train.csv
|   +-- financial_transaction_test.csv
|-- .gitattributes                    # Git LFS tracking rule for *.safetensors weights
|-- .gitignore                        # Excludes .venv/, __pycache__/, and temporary logs
|-- README.md                         # Project documentation
+-- requirements.txt                  # Cross-platform Python dependencies
