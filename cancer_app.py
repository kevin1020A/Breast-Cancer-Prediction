
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ============================================================
# BREAST CANCER AI - WEBSITE ONLY
# ============================================================
# This file DOES NOT train the model.
# It only:
#   1) loads cancer_model.joblib
#   2) loads the dataset for visualization
#   3) performs prediction
#   4) displays EDA / visualizations
#   5) compares the trained algorithms
# ============================================================

st.set_page_config(
    page_title="Breast Cancer AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cancer_model.joblib"
DATA_PATH = BASE_DIR / "Cancer_Data(1).csv"

# ============================================================
# Load model and dataset
# ============================================================

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH).drop(
        columns=["Unnamed: 32"],
        errors="ignore"
    )


if not MODEL_PATH.exists():
    st.error(
        "❌ cancer_model.joblib not found. "
        "Run cancer_model.py first."
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        "❌ Cancer_Data(1).csv not found. "
        "Put it in the same folder as this website."
    )
    st.stop()


artifact = load_artifact()
df = load_data()

model = artifact["model"]
features = artifact["feature_names"]
feature_ranges = artifact["feature_ranges"]
metrics = pd.DataFrame(artifact["metrics"]).T

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       FULL DARK MODE
       ======================================================== */

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: #0b1120 !important;
        color: #f8fafc !important;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Main text */
    h1, h2, h3, h4, h5, h6,
    p, label, span, div {
        color: #f8fafc;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }

    /* Hero */
    .hero {
        background: linear-gradient(
            135deg,
            #020617 0%,
            #111827 45%,
            #1e3a8a 100%
        );
        color: #ffffff !important;
        padding: 32px 38px;
        border-radius: 24px;
        margin-bottom: 25px;
        border: 1px solid #334155;
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
    }

    .hero h1,
    .hero p {
        color: #ffffff !important;
    }

    .hero p {
        color: #cbd5e1 !important;
    }

    /* Section headings */
    .section-title {
        color: #f8fafc !important;
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
    }

    .section-subtitle {
        color: #94a3b8 !important;
        margin-bottom: 20px;
    }

    /* Inputs */
    input,
    textarea,
    [data-baseweb="input"],
    [data-baseweb="select"] > div {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    input::placeholder {
        color: #64748b !important;
    }

    /* Selectbox / dropdown */
    [data-baseweb="select"] *,
    [role="option"] {
        color: #f8fafc !important;
        background-color: #111827 !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 16px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        color: #f8fafc !important;
    }

    /* Cards */
    .ready {
        background: #111827 !important;
        border: 2px dashed #475569;
        border-radius: 20px;
        padding: 35px;
        text-align: center;
        color: #cbd5e1 !important;
    }

    .ready h2,
    .ready p {
        color: #f8fafc !important;
    }

    /* Prediction cards */
    .result-benign {
        background: linear-gradient(
            135deg,
            #052e16,
            #064e3b
        );
        border: 2px solid #22c55e;
        border-radius: 20px;
        padding: 25px;
        color: #dcfce7 !important;
    }

    .result-benign h1,
    .result-benign h2,
    .result-benign p {
        color: #dcfce7 !important;
    }

    .result-malignant {
        background: linear-gradient(
            135deg,
            #450a0a,
            #7f1d1d
        );
        border: 2px solid #fb7185;
        border-radius: 20px;
        padding: 25px;
        color: #ffe4e6 !important;
    }

    .result-malignant h1,
    .result-malignant h2,
    .result-malignant p {
        color: #ffe4e6 !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 5px 18px rgba(0, 0, 0, .25);
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f8fafc !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: #111827 !important;
        border: 1px solid #334155 !important;
        border-radius: 14px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: #111827 !important;
        color: #f8fafc !important;
        border-color: #334155 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 700;
    }

    /* Dividers */
    hr {
        border-color: #334155 !important;
    }

    /* Markdown links */
    a {
        color: #60a5fa !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b !important;
        font-size: 13px;
        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Breast Cancer AI")
    st.caption("Machine Learning Dashboard")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🔮 Prediction",
            "📊 Data Visualization",
            "🏆 Algorithm Comparison",
        ],
    )

    st.divider()

    st.markdown("### Dataset")

    st.write(f"**Samples:** {len(df):,}")
    st.write(f"**Features:** {len(features)}")

    st.divider()

    st.markdown("### Deployed Model")
    st.write(f"**{artifact['model_name']}**")

# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🩺 Breast Cancer AI</h1>
        <p>
            Machine-learning dashboard for breast tumor classification,
            exploratory analysis, and model comparison.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PAGE 1 - PREDICTION
# ============================================================

if page == "🔮 Prediction":

    left, right = st.columns(
        [1.45, 1],
        gap="large"
    )

    inputs = {}

    with left:

        st.markdown(
            '<div class="section-title">Patient Measurements</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Enter the tumor measurements used by the trained model.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Mean features
        # ----------------------------------------------------

        mean_features = [
            f for f in features
            if f.endswith("_mean")
        ]

        st.markdown("#### 📏 Mean Measurements")

        cols = st.columns(2)

        for i, feature in enumerate(mean_features):

            r = feature_ranges[feature]

            label = (
                feature
                .replace("_mean", "")
                .replace("_", " ")
                .title()
            )

            with cols[i % 2]:

                inputs[feature] = st.number_input(
                    label,
                    min_value=float(r["min"]),
                    max_value=float(r["max"]),
                    value=float(r["median"]),
                    format="%.5f",
                )

        # ----------------------------------------------------
        # Advanced features
        # ----------------------------------------------------

        with st.expander(
            "⚙️ Advanced Measurements — SE & Worst"
        ):

            advanced = [
                f for f in features
                if not f.endswith("_mean")
            ]

            cols = st.columns(2)

            for i, feature in enumerate(advanced):

                r = feature_ranges[feature]

                with cols[i % 2]:

                    inputs[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=float(r["min"]),
                        max_value=float(r["max"]),
                        value=float(r["median"]),
                        format="%.5f",
                    )

        st.markdown("")

        predict = st.button(
            "🔍 Predict Diagnosis",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True,
        )

        if predict:

            input_df = pd.DataFrame(
                [[inputs[f] for f in features]],
                columns=features,
            )

            # Probability of class 1 = M = Malignant
            malignant_probability = float(
                model.predict_proba(input_df)[0][1]
            )

            benign_probability = (
                1 - malignant_probability
            )

            # 50% threshold
            if malignant_probability >= 0.50:

                st.markdown(
                    f"""
                    <div class="result-malignant">
                        <h1>⚠️ Malignant</h1>
                        <p>
                            Prediction: <b>Malignant (M)</b>
                        </p>
                        <h2>
                            {malignant_probability:.1%}
                        </h2>
                        <p>
                            Malignant probability
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-benign">
                        <h1>✅ Benign</h1>
                        <p>
                            Prediction: <b>Benign (B)</b>
                        </p>
                        <h2>
                            {benign_probability:.1%}
                        </h2>
                        <p>
                            Benign probability
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "🟢 Benign",
                    f"{benign_probability:.1%}",
                )

            with c2:
                st.metric(
                    "🔴 Malignant",
                    f"{malignant_probability:.1%}",
                )

            st.progress(
                malignant_probability,
                text=(
                    f"Malignant probability: "
                    f"{malignant_probability:.1%}"
                ),
            )

            st.caption(
                "Threshold: 50% malignant probability."
            )

            st.warning(
                "⚕️ Educational machine-learning project only — "
                "not a medical diagnosis."
            )

        else:

            st.markdown(
                """
                <div class="ready">
                    <h2>🔬 Ready for Prediction</h2>
                    <p>
                        Enter the measurements and click
                        <b>Predict Diagnosis</b>.
                    </p>
                    <br>
                    <p>
                        🟢 Benign: malignant probability &lt; 50%<br>
                        🔴 Malignant: malignant probability ≥ 50%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# PAGE 2 - DATA VISUALIZATION
# ============================================================

elif page == "📊 Data Visualization":

    st.markdown(
        '<div class="section-title">📊 Data Visualization</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore the dataset and understand the relationship between '
        'the features and diagnosis.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # 1. Diagnosis Distribution
    # ========================================================

    st.markdown("### 1️⃣ Diagnosis Distribution")

    c1, c2, c3 = st.columns(3)

    benign_count = int(
        (df["diagnosis"] == "B").sum()
    )

    malignant_count = int(
        (df["diagnosis"] == "M").sum()
    )

    c1.metric(
        "Total Samples",
        f"{len(df):,}"
    )

    c2.metric(
        "🟢 Benign",
        f"{benign_count:,}"
    )

    c3.metric(
        "🔴 Malignant",
        f"{malignant_count:,}"
    )

    counts = (
        df["diagnosis"]
        .value_counts()
        .rename(
            index={
                "B": "Benign",
                "M": "Malignant",
            }
        )
    )

    st.bar_chart(counts)

    # ========================================================
    # 2. Correlation Heatmap
    # ========================================================

    st.markdown("### 2️⃣ Feature Correlation Heatmap")

    st.caption(
        "This heatmap shows how strongly the numerical features "
        "are correlated with each other."
    )

    corr = df[features].corr()

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    heatmap = ax.imshow(
        corr,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="Correlation"
    )

    ax.set_title(
        "Feature Correlation Heatmap",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xticks(
        np.arange(len(corr.columns))
    )

    ax.set_yticks(
        np.arange(len(corr.columns))
    )

    ax.set_xticklabels(
        corr.columns,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        corr.columns,
        fontsize=7,
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ========================================================
    # 3. Feature Distributions
    # ========================================================

    st.markdown("### 3️⃣ Feature Distributions")

    selected_features = st.multiselect(
        "Choose features to visualize",
        features,
        default=[
            features[0],
            features[1],
            features[2],
            features[3],
        ],
    )

    if selected_features:

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8)
        )

        axes = axes.flatten()

        for i, feature in enumerate(
            selected_features[:4]
        ):

            axes[i].hist(
                df[feature],
                bins=25,
            )

            axes[i].set_title(feature)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")

        for j in range(
            len(selected_features[:4]),
            4
        ):
            axes[j].axis("off")

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # ========================================================
    # 4. Benign vs Malignant Mean Features
    # ========================================================

    st.markdown(
        "### 4️⃣ Benign vs Malignant Feature Comparison"
    )

    mean_features = [
        f for f in features
        if f.endswith("_mean")
    ]

    comparison = (
        df.groupby("diagnosis")[mean_features]
        .mean()
        .T
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    x = np.arange(
        len(mean_features)
    )

    width = 0.38

    ax.bar(
        x - width / 2,
        comparison["B"],
        width,
        label="Benign",
    )

    ax.bar(
        x + width / 2,
        comparison["M"],
        width,
        label="Malignant",
    )

    ax.set_title(
        "Mean Feature Comparison",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Feature")
    ax.set_ylabel("Average Value")

    ax.set_xticks(x)

    ax.set_xticklabels(
        mean_features,
        rotation=90,
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    # ========================================================
    # 5. Dataset Table
    # ========================================================

    st.markdown("### 5️⃣ Dataset Preview")

    st.dataframe(
        df.head(25),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# PAGE 3 - ALGORITHM COMPARISON
# ============================================================

elif page == "🏆 Algorithm Comparison":

    st.markdown(
        '<div class="section-title">'
        '🏆 Algorithm Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Compare the four algorithms evaluated during model development.'
        '</div>',
        unsafe_allow_html=True,
    )

    scores = metrics.sort_values(
        "F1 Score",
        ascending=False
    )

    st.success(
        f"🏆 Deployed model: **{artifact['model_name']}**"
    )

    # ========================================================
    # Metrics table
    # ========================================================

    st.markdown("### 📋 Performance Table")

    st.dataframe(
        scores.style.format("{:.2%}"),
        use_container_width=True,
    )

    # ========================================================
    # Accuracy
    # ========================================================

    st.markdown("### Accuracy")

    st.bar_chart(
        scores["Accuracy"]
    )

    # ========================================================
    # Precision
    # ========================================================

    st.markdown("### Precision")

    st.bar_chart(
        scores["Precision"]
    )

    # ========================================================
    # Recall
    # ========================================================

    st.markdown("### Recall")

    st.bar_chart(
        scores["Recall"]
    )

    # ========================================================
    # F1
    # ========================================================

    st.markdown("### F1 Score")

    st.bar_chart(
        scores["F1 Score"]
    )

    # ========================================================
    # All metrics together
    # ========================================================

    st.markdown(
        "### 📊 Complete Algorithm Comparison"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]

    x = np.arange(
        len(scores.index)
    )

    width = 0.20

    for i, metric_name in enumerate(
        metrics_to_plot
    ):

        ax.bar(
            x + (i - 1.5) * width,
            scores[metric_name],
            width,
            label=metric_name,
        )

    ax.set_title(
        "Algorithm Performance Comparison",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Score")

    ax.set_ylim(
        max(0, scores[metrics_to_plot].min().min() - 0.05),
        1.0,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        scores.index,
        rotation=20,
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.info(
        "The website does not retrain the algorithms. "
        "All training, preprocessing, scaling, and evaluation "
        "are performed in cancer_model.py."
    )

st.divider()

st.caption(
    "🩺 Breast Cancer AI • Machine Learning Project • Educational Use Only"
)
