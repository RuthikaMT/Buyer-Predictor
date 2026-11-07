"""
Buyer Predictor — Streamlit front-end for ML Assignment 2.
Loads the 5 pre-trained pipelines from model/ and evaluates them on an uploaded
test CSV. Nothing is retrained here.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from src.utils import FEATURES, MODEL_FILES, TARGET, compute_metrics, prepare_frame

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
ACCENT, MUTED = "#7C3AED", "#94A3B8"

st.set_page_config(page_title="Buyer Predictor", page_icon="🛒", layout="wide")

st.markdown(
    f"""
    <style>
      .block-container {{padding-top: 2rem;}}
      .hero {{border-left: 6px solid {ACCENT}; padding: .4rem 1rem; margin-bottom: 1rem;}}
      .hero h1 {{margin: 0; font-size: 2rem;}}
      .hero p {{margin: .2rem 0 0; color: #64748B;}}
      div[data-testid="stMetric"] {{border: 1px solid rgba(148,163,184,.35);
          border-radius: 10px; padding: .6rem .8rem;}}
    </style>
    <div class="hero">
      <h1>🛒 Buyer Predictor</h1>
      <p>Predicting which online visitors will buy, using 5 ML models</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


@st.cache_data
def load_reference_metrics():
    path = MODEL_DIR / "metrics.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


def evaluate(model, X, y, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    return pred, prob, compute_metrics(y, pred, prob)


models = load_models()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.download_button("⬇️ Download sample test_data.csv", sample.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    threshold = st.slider("Decision threshold (buyer if P ≥ threshold)", 0.05, 0.95, 0.50, 0.05)
    st.caption("Models were trained offline (see train_models.py). "
               "The app only loads them and scores your file.")

if uploaded is None:
    st.info("👈 Upload **test_data.csv** (or any CSV with the same columns incl. `Revenue`) to begin.")
    ref = load_reference_metrics()
    if ref is not None:
        st.subheader("Reference results on the held-out test split")
        st.dataframe(ref.style.format("{:.4f}").highlight_max(axis=0, color="#EDE9FE"),
                     width="stretch")
    with st.expander("Required columns"):
        st.write(", ".join(FEATURES + [TARGET]))
    st.stop()

# ---------------- Data ----------------
try:
    raw = pd.read_csv(uploaded)
    X, y = prepare_frame(raw)
except Exception as err:  # bad file -> friendly message instead of a stack trace
    st.error(f"Could not use this file: {err}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(raw):,}")
c2.metric("Buyers (Revenue = 1)", f"{int(y.sum()):,}")
c3.metric("Buyer rate", f"{y.mean():.1%}")

tab_single, tab_compare, tab_data = st.tabs(["🔍 Selected model", "📊 Compare all models", "🗂️ Data preview"])

# ---------------- Selected model ----------------
with tab_single:
    pred, prob, m = evaluate(models[model_name], X, y, threshold)
    st.subheader(f"{model_name} — evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax,
                    xticklabels=["No purchase", "Purchase"],
                    yticklabels=["No purchase", "Purchase"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch")
    with right:
        st.markdown("**ROC curve**")
        fpr, tpr, _ = roc_curve(y, prob)
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
        ax2.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {m['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], ls="--", color=MUTED)
        ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2, width="stretch")

    st.markdown("**Classification report**")
    report = classification_report(y, pred, target_names=["No purchase", "Purchase"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.3f}"), width="stretch")

    out = raw.copy()
    out["Predicted_Revenue"], out["Purchase_Probability"] = pred, prob.round(4)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[model_name].split('.')[0]}.csv")

# ---------------- All models ----------------
with tab_compare:
    rows = {name: evaluate(mdl, X, y, threshold)[2] for name, mdl in models.items()}
    table = pd.DataFrame(rows).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#EDE9FE"),
                 width="stretch")

    metric = st.radio("Plot metric", list(table.columns), index=5, horizontal=True)
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    order = table[metric].sort_values()
    ax3.barh(order.index, order.values,
             color=[ACCENT if n == order.idxmax() else MUTED for n in order.index])
    ax3.set_xlim(0, 1); ax3.set_xlabel(metric)
    for i, v in enumerate(order.values):
        ax3.text(v + 0.01, i, f"{v:.3f}", va="center")
    st.pyplot(fig3, width="stretch")
    st.success(f"Best {metric} on this file: **{table[metric].idxmax()}**")

with tab_data:
    st.dataframe(raw.head(50), width="stretch")
