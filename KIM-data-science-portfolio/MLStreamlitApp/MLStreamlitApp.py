import streamlit as st  # Import Streamlit for building interactive web apps
import pandas as pd  # Import pandas for data manipulation
from sklearn.model_selection import train_test_split  # For splitting dataset into training and test sets
from sklearn.linear_model import LinearRegression, LogisticRegression  # Import linear and logistic regression models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Import decision tree models
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Import KNN models
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, r2_score  # Import evaluation metrics
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualizations using seaborn

st.set_page_config(page_title="Interactive ML App", layout="wide")  # Configure the app's layout and title
st.title("🤖 Interactive Machine Learning Playground")  # Set the main title of the app

# Sidebar - User options
st.sidebar.header("1. Upload or Select Dataset")  # Add a sidebar header for dataset selection
data_source = st.sidebar.radio("Choose dataset source", ("Upload your own", "Use sample dataset"))  # Radio to choose dataset source

# If user chooses to upload a dataset
if data_source == "Upload your own":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")  # File uploader widget for CSV
    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # Load uploaded CSV into a DataFrame
    else:
        st.warning("Please upload a dataset.")  # Warn user if no file uploaded
        st.stop()  # Stop execution until file is uploaded

# If user chooses a sample dataset
else:
    sample_name = st.sidebar.selectbox("Sample datasets", ("Iris (Classification)", "Boston (Regression)"))  # Dropdown for sample dataset
    if "Iris" in sample_name:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)  # Load Iris dataset
        df = iris.frame  # Use the DataFrame version
    else:
        from sklearn.datasets import load_diabetes
        boston = load_diabetes(as_frame=True)  # Load Diabetes dataset (Boston is deprecated)
        df = boston.frame

st.write("### 📊 Preview of Dataset")  # Section title
st.dataframe(df.head())  # Show first few rows of the dataset

# Feature/target selection
st.sidebar.header("2. Select Features and Target")  # Sidebar header for feature selection
with st.sidebar:
    target = st.selectbox("Select target variable", df.columns)  # Select target column
    features = st.multiselect("Select feature columns", df.columns.drop(target))  # Select input features

    if not features:
        st.warning("Select at least one feature.")  # Warn if no features selected
        st.stop()  # Stop app if features are missing

    task_type = st.radio("Choose task", ("Regression", "Classification"))  # Choose task type

# Model selection and hyperparameters
st.sidebar.header("3. Model & Hyperparameters")  # Sidebar section for model settings
model_name = st.sidebar.selectbox("Choose model", 
    ["Linear Regression", "Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])  # Model options

# Hyperparameter sliders based on model type
params = {}
if "Tree" in model_name:
    params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)  # Tree depth slider
if "Neighbors" in model_name:
    params["n_neighbors"] = st.sidebar.slider("Number of Neighbors", 1, 15, 5)  # K value slider

# Data split
X = df[features]  # Features matrix
y = df[target]  # Target vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

# Model initialization based on user choice
if model_name == "Linear Regression":
    model = LinearRegression()
elif model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)  # Increase iterations for convergence
elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=params["max_depth"]) if task_type == "Classification" else DecisionTreeRegressor(max_depth=params["max_depth"])
elif model_name == "K-Nearest Neighbors":
    model = KNeighborsClassifier(n_neighbors=params["n_neighbors"]) if task_type == "Classification" else KNeighborsRegressor(n_neighbors=params["n_neighbors"])

# Model training
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Predict on test set

# Display model performance
st.header("📈 Model Performance")

# For regression tasks
if task_type == "Regression":
    score = r2_score(y_test, y_pred)  # R² score for regression
    st.write(f"**R² Score:** {score:.3f}")  # Show R²
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)  # Scatter plot of actual vs predicted
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)  # Display plot

# For classification tasks
else:
    acc = accuracy_score(y_test, y_pred)  # Accuracy metric
    prec = precision_score(y_test, y_pred, average="weighted")  # Weighted precision
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")

    # Show ROC curve if binary classification
    if len(y.unique()) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
        fpr, tpr, _ = roc_curve(y_test, y_proba)  # Compute ROC curve
        roc_auc = auc(fpr, tpr)  # Compute AUC
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")  # Diagonal line for reference
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)  # Show ROC curve

# Footer sidebar info
st.sidebar.markdown("---")
st.sidebar.info("Customize your experiment and compare results!")  # Helpful tip in sidebar