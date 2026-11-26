💳 FinSense AI — Automated Bank & UPI Transaction CategorizerCourse: CSB306 — Natural Language Processing (Project ID: NLP-18)Live Application: https://finsense-ai-categorizer.streamlit.app📌 OverviewDigital payment rails in India (UPI, NEFT, IMPS, POS, and ACH) produce noisy, unstructured transaction narrations such as:Swiggy order payment | Ref:0f8f77bb | Amount: INR 450.00
ACH/DEBIT/BAJAJ FINANCE EMI/99210
UPI/CR/991827/ZERODHA BROKING LTD
FinSense AI is an end-to-end, modular NLP pipeline that strips banking-rail clutter via regular expressions and classifies transactions into five core personal-finance categories: Food, Travel, EMI, Investment, and Shopping.🏗️ Project ArchitectureBank-Transaction-Categorizer-NLP/
├── backend/                          # Core NLP preprocessing, training & inference engine
│   ├── __init__.py                   # Package initializer for cross-directory imports
│   ├── preprocess.py                 # Regex noise scrubber (clean_transaction_text)
│   ├── train_models.py               # Reproducible training & evaluation benchmark script
│   ├── predictor.py                  # Inference API (load_models & predict_transaction)
│   └── artifacts/                    # Serialized models, metrics.json & confusion_matrix.png
│       ├── baseline_model.pkl        # TF-IDF (1,2-gram) + Logistic Regression pipeline
│       ├── distilbert_txn_model/     # Fine-tuned DistilBERT weights (tracked via Git LFS)
│       ├── confusion_matrix.png      # 1,000-sample test confusion matrix heatmap
│       ├── error_examples.csv        # Misclassified test samples log
│       └── metrics.json              # Evaluated Accuracy & Macro F1 benchmarks
├── frontend/                         # Interactive Streamlit SaaS dashboard & demo assets
│   ├── app.py                        # 3-tab UI (Live Tester, Batch Analyzer, Benchmarks)
│   └── sample_statement.csv          # 8-row sample bank statement for one-click testing
├── data/                             # Labeled Kaggle Indian Banking Transaction splits
│   ├── financial_transaction_train.csv
│   └── financial_transaction_test.csv
├── .gitattributes                    # Git LFS tracking rule for *.safetensors weights
├── .gitignore                        # Excludes .venv/, __pycache__/, and temporary logs
├── README.md                         # Project documentation
└── requirements.txt                  # Cross-platform Python dependencies
🧠 NLP Pipeline & Modeling ApproachRegex Noise Scrubber (backend/preprocess.py):Lowercases text and normalizes delimiters (/, |, -, _).Strips hexadecimal reference IDs (Ref:0f8f77bb), currency tokens (INR, Rs), standalone numbers, banking rail tags (upi, neft, pos, ach, dr, cr), and bank codes (hdfc, icici, sbi, axis, kotak).Classical Baseline (TF-IDF + Logistic Regression):Extracts unigram and bigram TF-IDF features (ngram_range=(1, 2)) and fits a multinomial Logistic Regression classifier.Modern Transformer (Fine-Tuned DistilBERT):Fine-tunes distilbert-base-uncased with a 5-class sequence classification head (max_length=32) for contextual merchant and intent classification.Human-in-the-Loop Confidence Guardrail:Includes a configurable Manual Review Threshold slider in the UI sidebar to flag ambiguous peer-to-peer transfers (e.g., UPI/DR/771209/RAHUL SHARMA/SBI) when confidence drops below the threshold.📊 Quantitative Benchmark (Held-Out Test Set — 1,000 Samples)Model ArchitectureTest SamplesAccuracy (%)Macro F1 (%)Classical Baseline (TF-IDF + Logistic Regression)1,000100.00%100.00%Modern Transformer (Fine-tuned DistilBERT)1,000100.00%100.00%🚀 Local Setup & Execution1. Clone the Repository (with Git LFS)git lfs install
git clone https://github.com/Shaiksyed097/Bank-Transaction-Categorizer-NLP.git
cd Bank-Transaction-Categorizer-NLP
2. Create a Virtual Environment & Install Dependenciespython -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
3. Launch the Streamlit Applicationstreamlit run frontend/app.py
(Optional) To retrain both models and regenerate evaluation artifacts from scratch:python -m backend.train_models
🖥️ Dashboard Features🔍 Live Transaction Tester: Test custom or preset UPI/bank narrations, inspect the intermediate Regex-cleaned signal, and compare TF-IDF vs. DistilBERT confidence distributions side-by-side.📊 Bank Statement Analyzer: Upload any CSV statement with a Transaction_Text column (or load the built-in demo CSV), view category spend KPIs and a Plotly donut chart, and export the categorized ledger.📈 Model Evaluation & Error Analysis: Inspect held-out Accuracy/Macro-F1 metrics, the 1,000-sample Confusion Matrix, and real-world edge-case notes.