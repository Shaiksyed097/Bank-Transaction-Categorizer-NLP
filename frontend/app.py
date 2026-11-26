import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow frontend/app.py to import cleanly from the sibling backend/ folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.predictor import ARTIFACTS_DIR, artifacts_exist, load_models, predict_transaction
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="FinSense AI • Bank & UPI Categorizer",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM FINTECH CSS STYLING ---
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0ea5e9 150%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-top: 0.4rem;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 10px;
    }
    .model-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    .model-card-highlight {
        background: linear-gradient(145deg, #1e293b, #0f2942);
        border: 1px solid #0ea5e9;
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.15);
    }
    .card-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0.3rem 0;
    }
    .card-conf {
        font-size: 0.9rem;
        color: #38bdf8;
        font-weight: 600;
    }
    .token-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 0.85rem 1.2rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        color: #e2e8f0;
        margin: 1rem 0 1.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

CATEGORY_ICONS = {
    "Food": "🍔 Food",
    "Travel": "✈️ Travel",
    "EMI": "🏦 EMI",
    "Investment": "📈 Investment",
    "Shopping": "🛍️ Shopping"
}


@st.cache_resource
def get_backend_engine():
    return load_models()


if not artifacts_exist():
    st.error("Backend artifacts not found! Run `python backend/train_models.py` first.")
    st.stop()

baseline_bundle, tokenizer, bert_model, metrics = get_backend_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ FinSense NLP Engine")
    st.caption("CSB306 • S.syed basha (12405575)")
    st.divider()
    conf_threshold = st.slider(
        "Manual Review Threshold (%)",
        min_value=40,
        max_value=95,
        value=60,
        step=5,
        help="Transactions below this DistilBERT confidence score are flagged for manual review."
    )
    st.divider()
    st.markdown("**Supported Categories**")
    for _, badge in CATEGORY_ICONS.items():
        st.markdown(f"- {badge}")
    st.divider()
    st.caption("Architecture: `frontend/app.py` → `backend/predictor.py` → DistilBERT")

# --- HERO HEADER ---
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">💳 FinSense AI — Bank & UPI Transaction Categorizer</div>
    <div class="hero-sub">Modular Frontend & Backend NLP Pipeline for Automated Financial Narration Classification</div>
    <div>
        <span class="badge-pill">Regex Noise Scrubber</span>
        <span class="badge-pill">TF-IDF + Logistic Regression</span>
        <span class="badge-pill">Fine-Tuned DistilBERT</span>
        <span class="badge-pill">CSB306 • NLP-18</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "🔍 Live Transaction Tester",
    "📊 Bank Statement Analyzer",
    "📈 Model Evaluation & Error Analysis"
])

# --- TAB 1: LIVE SINGLE TRANSACTION TESTER ---
with tab1:
    col_in, col_preset = st.columns([2.2, 1])
    samples = [
        "Swiggy order payment | Ref:0f8f77bb | Amount: INR 450.00",
        "Ola cab fare | Ref:8b98e0de | Amount: INR 320.50",
        "ACH/DEBIT/BAJAJ FINANCE EMI/99210",
        "UPI/CR/991827/ZERODHA BROKING LTD",
        "POS/MYNTRA DESIGNS PVT LTD/MUMBAI",
        "UPI/DR/771209/RAHUL SHARMA/SBI"
    ]
    with col_preset:
        selected = st.selectbox("Quick Sample Presets:", ["Custom Input"] + samples)
    default_txt = samples[0] if selected == "Custom Input" else selected
    with col_in:
        raw_input = st.text_input(
            "Enter Raw Bank / UPI Narration:",
            value=default_txt,
            placeholder="e.g., UPI/DR/98123/SWIGGY/ICICI"
        )

    if st.button("🚀 Analyze & Categorize", type="primary") and raw_input.strip():
        cleaned, (b_lbl, b_conf), (d_lbl, d_conf), prob_df = predict_transaction(
            raw_input, baseline_bundle, tokenizer, bert_model
        )

        st.markdown(
            f'<div class="token-box"><strong>🧹 Step 1 • Regex Cleaned Signal:</strong> &nbsp;<code>{cleaned if cleaned else "[empty after noise removal]"}</code></div>',
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="model-card">
                <div class="card-label">Classical Baseline • TF-IDF + LogReg</div>
                <div class="card-value">{CATEGORY_ICONS.get(b_lbl, b_lbl)}</div>
                <div class="card-conf">Confidence: {b_conf * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            needs_review = (d_conf * 100) < conf_threshold
            status_tag = "⚠️ Needs Manual Review" if needs_review else "✅ High Confidence"
            st.markdown(f"""
            <div class="model-card-highlight">
                <div class="card-label">Modern Transformer • Fine-Tuned DistilBERT</div>
                <div class="card-value">{CATEGORY_ICONS.get(d_lbl, d_lbl)}</div>
                <div class="card-conf">Confidence: {d_conf * 100:.1f}% &nbsp;•&nbsp; {status_tag}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Class Probability Distribution Comparison")
        fig_bar = px.bar(
            prob_df,
            x="Category",
            y="Confidence (%)",
            color="Model",
            barmode="group",
            color_discrete_map={"TF-IDF + LogReg": "#64748b", "DistilBERT": "#0ea5e9"},
            height=320
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 105],
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar, width="stretch")

# --- TAB 2: BATCH CSV STATEMENT ANALYZER ---
with tab2:
    col_up, col_demo = st.columns([2.5, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload Bank Statement CSV (`Transaction_Text` column required)", type=["csv"])
    with col_demo:
        st.write("")
        st.write("")
        load_demo = st.button("📂 Load Demo Statement CSV", width="stretch")

    sample_csv_path = os.path.join(FRONTEND_DIR, "sample_statement.csv")
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif load_demo and os.path.exists(sample_csv_path):
        df = pd.read_csv(sample_csv_path)

    if df is not None:
        if "Transaction_Text" not in df.columns:
            st.error("CSV file must contain a column named `Transaction_Text`.")
        else:
            results = [predict_transaction(t, baseline_bundle, tokenizer, bert_model) for t in df["Transaction_Text"]]
            df["Cleaned_Text"] = [r[0] for r in results]
            df["Baseline_Pred"] = [r[1][0] for r in results]
            df["Category"] = [r[2][0] for r in results]
            df["Confidence_%"] = [round(r[2][1] * 100, 1) for r in results]

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Transactions", len(df))
            if "Amount" in df.columns:
                k2.metric("Total Spend", f"₹{df['Amount'].sum():,.0f}")
                top_cat = df.groupby("Category")["Amount"].sum().idxmax()
            else:
                k2.metric("Avg Confidence", f"{df['Confidence_%'].mean():.1f}%")
                top_cat = df["Category"].value_counts().idxmax()
            k3.metric("Top Spend Category", CATEGORY_ICONS.get(top_cat, top_cat))
            flagged = int((df["Confidence_%"] < conf_threshold).sum())
            k4.metric("Flagged for Review", flagged)

            st.divider()

            left, right = st.columns([1.35, 1])
            with left:
                st.markdown("**Categorized Ledger**")
                st.dataframe(df, width="stretch", height=360)
            with right:
                st.markdown("**Expense Breakdown**")
                if "Amount" in df.columns:
                    fig = px.pie(
                        df, names="Category", values="Amount", hole=0.48,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                else:
                    fig = px.pie(
                        df, names="Category", hole=0.48,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=360
                )
                st.plotly_chart(fig, width="stretch")

            st.download_button(
                "⬇️ Export Categorized Statement (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                "categorized_statement.csv",
                "text/csv",
                type="primary"
            )

# --- TAB 3: MODEL BENCHMARKS & ERROR ANALYSIS ---
with tab3:
    st.subheader("Quantitative Model Benchmark (Held-Out Test Split)")
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model Architecture"})
    metrics_df["Accuracy (%)"] = metrics_df["Accuracy (%)"].map(lambda x: f"{x:.2f}%")
    metrics_df["Macro F1 (%)"] = metrics_df["Macro F1 (%)"].map(lambda x: f"{x:.2f}%")
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.divider()

    c_img, c_err = st.columns([1, 1.15])
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    err_path = os.path.join(ARTIFACTS_DIR, "error_examples.csv")

    with c_img:
        st.markdown("#### Confusion Matrix (1,000 Test Samples)")
        if os.path.exists(cm_path):
            st.image(cm_path, width="stretch")
    with c_err:
        st.markdown("#### Edge-Case & Error Analysis")
        if os.path.exists(err_path):
            err_df = pd.read_csv(err_path)
            if err_df.empty:
                st.success("✅ **0 misclassifications on the held-out test set (100.00% Accuracy).**")
                st.info(
                    "**Viva Edge-Case Discussion:**\n"
                    "- Because the synthetic test split uses the same core merchant vocabulary (`Swiggy`, `Ola`, `Zerodha`, `Myntra`) as training, both TF-IDF and DistilBERT achieve 100% accuracy.\n"
                    "- **Real-World Failure Modes:** Ambiguous peer-to-peer UPI transfers (e.g., `UPI/DR/771209/RAHUL SHARMA/SBI`) or multi-purpose marketplaces (`Amazon Pay`) lack explicit category cues, which is why our UI implements a **Confidence Threshold Review Flag** in the sidebar."
                )
            else:
                st.dataframe(err_df, width="stretch")