"""
Stellar Object Classification - Streamlit App
-----------------------------------------------
Classifies SDSS17 astronomical observations as STAR, GALAXY, or QSO
based on photometric and spectroscopic features, using an XGBoost model
trained on the 'star_classification.csv' dataset.

To deploy on Streamlit Community Cloud:
1. Push this file, `star_classification.csv`, and `requirements.txt`
   to a GitHub repository.
2. On https://share.streamlit.io create a new app pointing at this file.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Stellar Object Classification",
    page_icon="🌌",
    layout="wide",
)

DATA_PATH = "star_classification.csv"

ID_COLUMNS = [
    "obj_ID", "run_ID", "rerun_ID", "cam_col",
    "field_ID", "plate", "MJD", "fiber_ID", "spec_obj_ID"
]

BASE_FEATURES = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

CLASS_NAMES = {0: "GALAXY", 1: "QSO", 2: "STAR"}  # overwritten after fit with real mapping


# --------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop identifier columns that carry no physical/predictive meaning
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])

    # Remove sentinel/error values (SDSS uses extreme negative magnitudes,
    # e.g. -9999, to flag bad photometric measurements)
    for col in ["u", "g", "z"]:
        df = df[df[col] > -1000]

    # Feature engineering: photometric colour indices
    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]
    df["u_r"] = df["u"] - df["r"]
    df["g_i"] = df["g"] - df["i"]

    return df


def engineer_row(alpha, delta, u, g, r, i, z, redshift) -> pd.DataFrame:
    """Build a single-row feature frame matching the training schema."""
    row = {
        "alpha": alpha, "delta": delta,
        "u": u, "g": g, "r": r, "i": i, "z": z,
        "redshift": redshift,
        "u_g": u - g,
        "g_r": g - r,
        "r_i": r - i,
        "i_z": i - z,
        "u_r": u - r,
        "g_i": g - i,
    }
    return pd.DataFrame([row])


# --------------------------------------------------------------------------
# Model training (cached so it only runs once per deployment)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training XGBoost model...")
def train_model(df: pd.DataFrame):
    X = df.drop(columns="class")
    y_raw = df["class"]

    classes = sorted(y_raw.unique())
    class_to_idx = {c: idx for idx, c in enumerate(classes)}
    idx_to_class = {idx: c for c, idx in class_to_idx.items()}
    y = y_raw.map(class_to_idx)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
    }

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, idx_to_class, metrics, feature_importance, list(X.columns)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.title("🌌 Stellar Object Classification")
st.caption(
    "Classify SDSS17 observations as a **Star**, **Galaxy**, or **Quasar (QSO)** "
    "using an XGBoost model trained on photometric and spectroscopic data."
)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Please make sure the dataset CSV file "
        "is included in the same directory as this app (and in your GitHub repo)."
    )
    st.stop()

model, idx_to_class, metrics, feature_importance, feature_cols = train_model(df)

tab_predict, tab_batch, tab_performance, tab_about = st.tabs(
    ["🔭 Single Prediction", "📁 Batch Prediction", "📊 Model Performance", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Tab 1: Single prediction
# --------------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter Observation Details")

    col1, col2 = st.columns(2)

    with col1:
        alpha = st.number_input("Alpha (Right Ascension)", value=180.0, format="%.6f")
        delta = st.number_input("Delta (Declination)", value=30.0, format="%.6f")
        u = st.number_input("u (Ultraviolet magnitude)", value=19.5, format="%.4f")
        g = st.number_input("g (Green magnitude)", value=18.0, format="%.4f")

    with col2:
        r = st.number_input("r (Red magnitude)", value=17.5, format="%.4f")
        i = st.number_input("i (Near-infrared magnitude)", value=17.0, format="%.4f")
        z = st.number_input("z (Infrared magnitude)", value=16.7, format="%.4f")
        redshift = st.number_input("Redshift", value=0.1, format="%.6f")

    if st.button("Predict Object Class", type="primary"):
        input_df = engineer_row(alpha, delta, u, g, r, i, z, redshift)
        input_df = input_df[feature_cols]  # ensure correct column order

        pred_idx = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        pred_class = idx_to_class[pred_idx]

        st.success(f"Predicted class: **{pred_class}**")

        proba_df = pd.DataFrame({
            "Class": [idx_to_class[idx] for idx in range(len(pred_proba))],
            "Probability": pred_proba
        }).sort_values("Probability", ascending=False)

        fig = px.bar(
            proba_df, x="Class", y="Probability", color="Class",
            text_auto=".2%", title="Prediction Confidence"
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 2: Batch prediction via CSV upload
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV for Batch Predictions")
    st.write(
        "The file must contain the following columns: "
        f"`{', '.join(BASE_FEATURES)}`"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        missing_cols = [c for c in BASE_FEATURES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            engineered = batch_df.copy()
            engineered["u_g"] = engineered["u"] - engineered["g"]
            engineered["g_r"] = engineered["g"] - engineered["r"]
            engineered["r_i"] = engineered["r"] - engineered["i"]
            engineered["i_z"] = engineered["i"] - engineered["z"]
            engineered["u_r"] = engineered["u"] - engineered["r"]
            engineered["g_i"] = engineered["g"] - engineered["i"]

            preds = model.predict(engineered[feature_cols])
            batch_df["Predicted_Class"] = [idx_to_class[p] for p in preds]

            st.write(batch_df)

            csv_out = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions as CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv",
            )

# --------------------------------------------------------------------------
# Tab 3: Model performance
# --------------------------------------------------------------------------
with tab_performance:
    st.subheader("Model Evaluation Metrics (Held-out Test Set)")

    metric_cols = st.columns(4)
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(name, f"{value:.2%}")

    st.subheader("Feature Importance")
    fig_imp = px.bar(
        feature_importance, x="Importance", y="Feature",
        orientation="h", title="XGBoost Feature Importance"
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Class Distribution in Training Data")
    class_counts = df["class"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    fig_class = px.bar(class_counts, x="Class", y="Count", color="Class")
    st.plotly_chart(fig_class, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 4: About
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("About this Project")
    st.markdown(
        """
        This app classifies astronomical objects observed by the
        **Sloan Digital Sky Survey (SDSS17)** into three categories:

        - **STAR**
        - **GALAXY**
        - **QSO** (Quasar)

        **Model:** XGBoost Classifier
        **Key features used:** photometric magnitudes (u, g, r, i, z),
        sky coordinates (alpha, delta), redshift, and engineered
        colour-index features (u−g, g−r, r−i, i−z, u−r, g−i).

        Redshift is the single most informative feature: stars have
        redshift values very close to zero, galaxies show moderate
        redshift, and quasars exhibit much higher redshift values.
        """
    )
