import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="Bankruptcy Prevention Predictor",
    page_icon="🏦",
    layout="wide"
)

FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk"
]

DISPLAY_NAMES = {
    "industrial_risk": "Industrial Risk",
    "management_risk": "Management Risk",
    "financial_flexibility": "Financial Flexibility",
    "credibility": "Credibility",
    "competitiveness": "Competitiveness",
    "operating_risk": "Operating Risk"
}


@st.cache_data
def load_data():
    df = pd.read_csv(
        "Bankruptcy_prevention_Dataset.csv",
        sep=";"
    )
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def train_models(df):
    X = df[FEATURES].copy()
    y = df["class"].map({
        "bankruptcy": 1,
        "non-bankruptcy": 0
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        )
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred)
        })

    return (
        models,
        pd.DataFrame(results),
        X_test,
        y_test
    )


try:
    df = load_data()
    models, results_df, X_test, y_test = train_models(df)
except FileNotFoundError:
    st.error(
        "Dataset not found. Keep "
        "'Bankruptcy_prevention_Dataset(4).csv' "
        "in the same folder as app.py."
    )
    st.stop()


st.title("🏦 Bankruptcy Prevention Prediction")
st.caption(
    "Machine Learning Classification Application"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Prediction",
        "📊 Model Performance",
        "📈 Dataset Analysis",
        "ℹ️ About"
    ]
)


if page == "🏠 Home":

    st.header("Project Overview")

    st.write(
        "This application predicts whether a company is classified "
        "as bankruptcy or non-bankruptcy using six business-risk "
        "features."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Original Records", 250)
    c2.metric("Unique Records", len(df))
    c3.metric("Features", 6)
    c4.metric("Models", 4)

    st.subheader("Input Features")

    feature_table = pd.DataFrame({
        "Feature": FEATURES,
        "Description": [DISPLAY_NAMES[x] for x in FEATURES]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Target Distribution")

    st.bar_chart(df["class"].value_counts())


elif page == "🔮 Prediction":

    st.header("Company Bankruptcy Prediction")

    st.info(
        "The dataset uses risk scores 0, 0.5 and 1."
    )

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        index=3
    )

    values = {}

    c1, c2 = st.columns(2)

    for i, feature in enumerate(FEATURES):
        with c1 if i % 2 == 0 else c2:
            values[feature] = st.select_slider(
                DISPLAY_NAMES[feature],
                options=[0.0, 0.5, 1.0],
                value=0.5
            )

    input_df = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    st.subheader("Input Values")
    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "🔍 Predict Bankruptcy",
        type="primary",
        use_container_width=True
    ):

        model = models[selected_model]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error(
                "⚠️ Bankruptcy Predicted"
            )
        else:
            st.success(
                "✅ Non-Bankruptcy Predicted"
            )

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0]

            st.subheader("Prediction Probability")

            p1, p2 = st.columns(2)

            p1.metric(
                "Non-Bankruptcy",
                f"{probability[0] * 100:.2f}%"
            )

            p2.metric(
                "Bankruptcy",
                f"{probability[1] * 100:.2f}%"
            )

            st.progress(
                float(probability[1]),
                text=f"Bankruptcy probability: "
                     f"{probability[1] * 100:.2f}%"
            )


elif page == "📊 Model Performance":

    st.header("Model Performance")

    display_results = results_df.copy()
    display_results["Accuracy"] = (
        display_results["Accuracy"] * 100
    ).round(2)

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        display_results.set_index("Model")["Accuracy"]
    )

    st.subheader("Classification Reports")

    for name, model in models.items():

        pred = model.predict(X_test)

        report = classification_report(
            y_test,
            pred,
            target_names=[
                "Non-bankruptcy",
                "Bankruptcy"
            ],
            output_dict=True
        )

        st.write(f"### {name}")

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

        cm = confusion_matrix(
            y_test,
            pred
        )

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        ax.imshow(cm)

        ax.set_title(
            f"{name} - Confusion Matrix"
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["Non-bankruptcy", "Bankruptcy"]
        )

        ax.set_yticklabels(
            ["Non-bankruptcy", "Bankruptcy"]
        )

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        st.pyplot(fig)
        plt.close(fig)


elif page == "📈 Dataset Analysis":

    st.header("Dataset Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Dataset",
        "Statistics",
        "Feature Importance"
    ])

    with tab1:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            f"250 original records → "
            f"{len(df)} unique records after duplicate removal."
        )

    with tab2:

        st.dataframe(
            df[FEATURES].describe().round(3),
            use_container_width=True
        )

        selected_feature = st.selectbox(
            "Select Feature",
            FEATURES
        )

        counts = (
            df[selected_feature]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(counts)

    with tab3:

        rf = models["Random Forest"]

        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": rf.feature_importances_
        }).sort_values(
            "Importance",
            ascending=False
        )

        st.dataframe(
            importance.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            importance.set_index("Feature")
        )


else:

    st.header("ℹ️ About")

    st.write(
        """
        ### Bankruptcy Prevention

        This project predicts bankruptcy/non-bankruptcy using:

        - Industrial Risk
        - Management Risk
        - Financial Flexibility
        - Credibility
        - Competitiveness
        - Operating Risk

        Models used:

        - Logistic Regression
        - K-Nearest Neighbors
        - Decision Tree
        - Random Forest

        The application follows the uploaded notebook's workflow:
        duplicate removal, target encoding, stratified train/test
        split, scaling for Logistic Regression/KNN, model training,
        evaluation and prediction.
        """
    )

    st.warning(
        "The final dataset is small after duplicate removal, so "
        "model performance should be interpreted cautiously."
    )
