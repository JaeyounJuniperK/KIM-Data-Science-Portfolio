import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Interactive ML App", layout="wide")
st.title("🤖 Interactive Machine Learning Playground")

# Sidebar - User options
st.sidebar.header("1. Upload or Select Dataset")
data_source = st.sidebar.radio("Choose dataset source", ("Upload your own", "Use sample dataset"))

if data_source == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a dataset.")
        st.stop()
else:
    sample_name = st.sidebar.selectbox("Sample datasets", ("Iris (Classification)", "Boston (Regression)"))
    if "Iris" in sample_name:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
    else:
        from sklearn.datasets import load_diabetes
        boston = load_diabetes(as_frame=True)
        df = boston.frame

st.write("### 📊 Preview of Dataset")
st.dataframe(df.head())

# Feature/target selection
st.sidebar.header("2. Select Features and Target")
with st.sidebar:
    target = st.selectbox("Select target variable", df.columns)
    features = st.multiselect("Select feature columns", df.columns.drop(target))

    if not features:
        st.warning("Select at least one feature.")
        st.stop()

    task_type = st.radio("Choose task", ("Regression", "Classification"))

# Model selection and hyperparams
st.sidebar.header("3. Model & Hyperparameters")
model_name = st.sidebar.selectbox("Choose model", 
    ["Linear Regression", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

# Hyperparameter widgets
params = {}
if "Tree" in model_name:
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
if "Neighbors" in model_name:
    params["n_neighbors"] = st.sidebar.slider("Number of Neighbors", 1, 15, 5)

# Data split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=params["max_depth"]) if task_type == "Classification" else DecisionTreeRegressor(max_depth=params["max_depth"])
elif model_name == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=params["n_neighbors"]) if task_type == "Classification" else KNeighborsRegressor(n_neighbors=params["n_neighbors"])

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Performance metrics
st.header("📈 Model Performance")

if task_type == "Regression":
    score = r2_score(y_test, y_pred)
    st.write(f"**R² Score:** {score:.3f}")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

else:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")

    # ROC Curve
    if len(y.unique()) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Customize your experiment and compare results!")