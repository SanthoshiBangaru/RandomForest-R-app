import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RF Classifier",
    page_icon="🩺",
    layout="wide"
)

def load_css(css_file):

    with open(css_file) as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True

        )

load_css("style.css")

# =========================================================
# TITLE
# =========================================================

st.title("🩺 Random Forest Classifier Dashboard")

st.markdown("""
This application demonstrates Random Forest Classification
with hyperparameter tuning and preprocessing.

### Features Included
- Dataset Loading
- Data Preprocessing
- Hyperparameter Tuning
- Model Training
- Prediction
- Evaluation
- Feature Importance
- Save Preprocessed Dataset
""")

# =========================================================
# LOAD DATASET
# =========================================================

st.header("📂 Step 1 : Load Dataset")

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(
    data.target,
    name="target"
)

df = pd.concat([X, y], axis=1)

st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    use_container_width=True
)

# =========================================================
# DATASET INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Classes", y.nunique())

# =========================================================
# DATA CLEANING
# =========================================================

st.header("🧹 Step 2 : Data Cleaning")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

with col2:

    st.subheader("Duplicate Rows")

    duplicates = df.duplicated().sum()

    st.write(f"Duplicate Rows : {duplicates}")

# Remove duplicates

df.drop_duplicates(inplace=True)

st.success("✅ Data Cleaning Completed")

# =========================================================
# PREPROCESSING
# =========================================================

st.header("⚙️ Step 3 : Data Preprocessing")

X = df.drop("target", axis=1)
y = df["target"]

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

# =========================================================
# SAVE PREPROCESSED DATA
# =========================================================

st.subheader("💾 Save Preprocessed Dataset")

# Create folder if not exists

os.makedirs("data", exist_ok=True)

# Add target column

processed_df = X_scaled_df.copy()

processed_df["target"] = y.values

if st.button("Save Preprocessed Data"):

    processed_df.to_csv(
        "data/preprocessed_breast_cancer.csv",
        index=False
    )

    st.success(
        "✅ Preprocessed dataset saved successfully in data/ folder"
    )

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

st.header("✂️ Step 4 : Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Training Samples", X_train.shape[0])

with col2:
    st.metric("Testing Samples", X_test.shape[0])

# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

st.header("🎛️ Step 5 : Hyperparameter Tuning")

params = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

st.success("✅ Hyperparameter Tuning Completed")

# =========================================================
# BEST PARAMETERS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Best Estimators",
        grid.best_params_["n_estimators"]
    )

with col2:
    st.metric(
        "Best Depth",
        str(grid.best_params_["max_depth"])
    )

with col3:
    st.metric(
        "CV Accuracy",
        round(grid.best_score_, 4)
    )

# =========================================================
# MODEL TRAINING
# =========================================================

st.header("🤖 Step 6 : Model Training")

model.fit(X_train, y_train)

st.success("✅ Random Forest Model Trained Successfully")

# =========================================================
# PREDICTIONS
# =========================================================

st.header("📌 Step 7 : Predictions")

y_pred = model.predict(X_test)

prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.dataframe(
    prediction_df.head(10),
    use_container_width=True
)

# =========================================================
# MODEL EVALUATION
# =========================================================

st.header("📉 Step 8 : Model Evaluation")

acc = accuracy_score(y_test, y_pred)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        round(acc, 4)
    )

with col2:
    st.metric(
        "Error Rate",
        round(1 - acc, 4)
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    columns=["Predicted 0", "Predicted 1"],
    index=["Actual 0", "Actual 1"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df,
    use_container_width=True
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.header("📊 Step 9 : Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 10 Important Features"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# INSIGHTS
# =========================================================

st.header("🧠 Insights")

st.write("""
- Random Forest reduces overfitting using multiple decision trees.
- Ensemble learning improves classification accuracy.
- Hyperparameter tuning improves model performance.
- Feature importance helps identify influential medical attributes.
- Preprocessed dataset can be reused for future ML projects.
""")