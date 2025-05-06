import streamlit as st  # Import Streamlit library to create the web app
import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np  # Import NumPy for numerical operations
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
from sklearn.cluster import KMeans  # Import KMeans algorithm for clustering
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction
from sklearn.metrics import silhouette_score  # Import silhouette_score for model evaluation
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import seaborn as sns  # Import Seaborn for creating visualizations
from sklearn.datasets import load_iris, load_wine  # Import sample datasets (Iris and Wine) from scikit-learn

# Streamlit page setup
st.set_page_config(page_title="Unsupervised ML Explorer", layout="wide")  # Configure the page layout and title for the Streamlit app
st.title("🔍 Unsupervised Machine Learning Explorer")  # Set the title of the app

# Display introductory markdown text explaining the app's purpose and datasets
st.markdown(
    """
    This app allows you to explore **unsupervised machine learning** with your own datasets or sample data.
    Upload a dataset, adjust clustering settings, and visualize the results instantly.
    
    💐 The Iris dataset consists of 150 samples, with 50 samples of each species and is used especially for classification and clustering tasks. 
    Each sample contains four features (attributes) measured from the flowers: 
    - Sepal Length: The length of the sepal (the leaf-like structure that encloses the flower bud). 
    - Sepal Width: The width of the sepal. 
    - Petal Length: The length of the petal (the colorful, often fragrant part of the flower). 
    - Petal Width: The width of the petal. 

    The dataset is often used to perform classification tasks, where the goal is to predict the species of a flower based on these four features. 
    However, the dataset is also commonly used for clustering (as an unsupervised learning task) to see if the flowers naturally group into clusters that correspond to the species.
    The Iris dataset is often used to demonstrate supervised classification algorithms (e.g., decision trees, KNN, SVM).
    The dataset is also a good candidate for clustering algorithms like K-Means or hierarchical clustering, where no labels are provided, and the goal is to group the flowers based on their similarities. 

    🍷 The Wine dataset is based on chemical analysis of wines grown in the same region in Italy. It contains the results of a study conducted by the Italian Institute of Agronomy to classify wines based on chemical properties.
    The dataset consists of 178 samples, with 13 features for each sample:
    - Alcohol: Alcohol content in the wine.
    - Malic Acid: Amount of malic acid, which affects the wine's flavor.
    - Ash: Ash content, which can influence the color and flavor.
    - Alcalinity of Ash: pH-related property.
    - Magnesium: Magnesium content, which affects wine texture.
    - Total Phenols: Chemical compounds that contribute to wine color and bitterness.
    - Flavanoids: Compounds that contribute to color, bitterness, and taste.
    - Non Flavonoid Phenols: Chemical compounds also related to wine bitterness and flavor.
    - Proanthocyanins: Compounds that affect wine color and astringency.
    - Color Intensity: Measurement of color intensity.
    - Hue: The color tone of the wine.
    - OD280/OD315: The ratio of absorbance at two wavelengths (280 and 315 nm), used to assess the quality of wine.
    - Proline: An amino acid, which can indicate the overall quality of the wine.
    
    These features are used to predict and determine one of three wine types, each from different grape varieties. 
    """
)

# Sidebar for controls: Sidebar will allow users to upload a dataset or select a sample dataset
with st.sidebar:
    st.header("📁 Step 1: Upload or Select a Dataset")  # Sidebar header for dataset selection
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])  # Allow the user to upload a CSV file

    # Option to select a sample dataset (Iris or Wine)
    sample_option = st.selectbox("Or select a sample dataset", ["None", "Iris", "Wine"])  
    st.markdown("---")  # Horizontal line for separation in the sidebar
    st.header("⚙️ Step 2: Configure Model Settings")  # Sidebar header for model configuration

# Load dataset function that loads a sample dataset based on user choice
def load_sample_dataset(name):
    if name == "Iris":
        data = load_iris()  # Load Iris dataset
        return pd.DataFrame(data.data, columns=data.feature_names)  # Convert to DataFrame
    elif name == "Wine":
        data = load_wine()  # Load Wine dataset
        return pd.DataFrame(data.data, columns=data.feature_names)  # Convert to DataFrame
    return None  # Return None if no dataset is selected

# Check if a file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)  # Read the uploaded CSV file into a DataFrame
    source = "uploaded"  # Set source to "uploaded"
elif sample_option != "None":
    df = load_sample_dataset(sample_option)  # Load sample dataset based on user choice
    source = "sample"  # Set source to "sample"
else:
    st.warning("Please upload a dataset or select a sample dataset from the sidebar.")  # Show warning if no dataset is uploaded or selected
    st.stop()  # Stop the execution of the app if no dataset is provided

# Display a preview of the dataset to the user
st.subheader("📄 Dataset Preview")  
st.dataframe(df.head(), use_container_width=True)  # Display the first 5 rows of the dataset

# Feature selection with guidance: Allow user to select features for clustering
with st.sidebar:
    st.subheader("🔍 Feature Selection")  # Sidebar header for feature selection
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()  # Get numeric columns from the dataset
    selected_features = st.multiselect(  # Allow the user to select features for clustering
        "Select features to include in clustering:", 
        numeric_columns, 
        default=numeric_columns  # Default to all numeric columns
    )
    normalize = st.checkbox("Apply normalization (StandardScaler)", value=True)  # Option to apply normalization
    num_clusters = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1)  # Slider to select number of clusters

# Provide guidance for feature selection
st.markdown("""
### **Feature Selection Guidance**:
- Select at least **two numeric features** for clustering.
- Ensure that the features you select are relevant to the patterns you want to explore.
- For example, if you are clustering wines, you might select **alcohol, hue, and flavanoids**.
- Unrelated or highly correlated features may distort the clustering results, so use judgment when selecting features.
""")

# Ensure that at least two features are selected for clustering
if len(selected_features) < 2:
    st.error("Please select at least two numeric features for clustering.")  # Show error if less than two features are selected
    st.stop()  # Stop execution if feature selection is invalid

# Prepare data for clustering (apply normalization if selected)
X = df[selected_features].copy()  # Select the columns for clustering
if normalize:
    X = StandardScaler().fit_transform(X)  # Apply standard scaling if selected
else:
    X = X.values  # Use original values if normalization is not applied

# Apply KMeans clustering algorithm
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Initialize KMeans with the selected number of clusters
cluster_labels = kmeans.fit_predict(X)  # Perform clustering and assign cluster labels
df['Cluster'] = cluster_labels  # Add the cluster labels to the DataFrame

# Compute the silhouette score to evaluate clustering quality
sil_score = silhouette_score(X, cluster_labels)

# Apply PCA for visualization (reduce the dataset to 2 components for plotting)
pca = PCA(n_components=2)  # Initialize PCA with 2 components
components = pca.fit_transform(X)  # Apply PCA to the data
df['PC1'], df['PC2'] = components[:, 0], components[:, 1]  # Assign the two principal components to the DataFrame

# Display KMeans explanation
st.subheader("🔑 Model Explanation: K-Means Clustering")  
st.markdown("""
K-Means is an unsupervised learning algorithm that groups data points into a predefined number of clusters (k). It works by:
- Assigning each data point to the nearest cluster center (centroid).
- Recomputing centroids based on the mean of the points assigned to each cluster.
- Repeating the process until the centroids stabilize.

**Cluster Centers** (Centroids): 
These represent the average position of points within each cluster. You can interpret the centroid values as the "center" of the cluster.
""")

# Display centroids (cluster centers)
st.subheader("🧑‍🏫 Cluster Centers (Centroids)")  
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=selected_features)  # Create DataFrame for centroids
st.write(centroids)  # Display centroids in the app

# Tabs for visualization and model performance feedback
tab1, tab2, tab3 = st.tabs(["📊 Cluster Plot", "📈 Elbow Plot", "📐 Silhouette Score"])  # Create tabs for different visualizations

# Tab 1: Cluster Plot (PCA Projection)
with tab1:
    st.subheader("Cluster Visualization (PCA Projection)")  # Set subtitle for the tab
    fig1, ax1 = plt.subplots()  # Create a Matplotlib figure
    sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set2', ax=ax1, s=60)  # Scatter plot of PCA components
    ax1.set_title(f"K-Means Clustering with k={num_clusters}")  # Set title for the plot
    st.pyplot(fig1)  # Display the plot in Streamlit
    st.markdown("""
    **Cluster Plot (PCA Projection)**:
    The plot shows how the data points are grouped into clusters in 2D. Each point is colored based on its assigned cluster. The axes represent the **two principal components** that capture the most variance in the data.

    **Interpretation**: 
    - Points in the same color belong to the same cluster.
    - The farther apart the clusters are, the better they are separated.
    """)

# Tab 2: Elbow Plot (Optimal k)
with tab2:
    st.subheader("Elbow Method for Optimal k")  # Set subtitle for the tab
    inertia_values = []  # List to store inertia values
    K_range = range(1, 11)  # Range for k (number of clusters)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)  # Initialize KMeans for different k values
        km.fit(X)  # Fit the model
        inertia_values.append(km.inertia_)  # Store inertia value (sum of squared distances from points to centroids)
    fig2, ax2 = plt.subplots()  # Create figure for Elbow plot
    ax2.plot(K_range, inertia_values, marker='o')  # Plot inertia values
    ax2.set_xlabel('Number of clusters (k)')  # Label for x-axis
    ax2.set_ylabel('Inertia')  # Label for y-axis
    ax2.set_title("Elbow Plot")  # Set title for the plot
    st.pyplot(fig2)  # Display the plot

    st.markdown("""
    **Elbow Plot**:
    The **Elbow Method** helps determine the optimal number of clusters (k). The inertia represents the sum of squared distances from each point to its assigned cluster center.

    **Interpretation**:
    - Look for the "elbow" in the plot, where inertia starts to decrease more slowly.
    - The optimal k is usually at the point where the inertia reduction slows down.

    **Tip**: If the elbow is not clearly visible, experiment with different k values to find a suitable choice.
    """)

# Tab 3: Silhouette Score
with tab3:
    st.subheader("Silhouette Score")  # Set subtitle for the tab
    st.markdown(f"""
    **Silhouette Score** measures how similar data points are to their own cluster compared to other clusters. It ranges from -1 to 1, with higher values indicating better-defined clusters.

    A score close to **1** means the points are well clustered. A score close to **0** means the clusters overlap. A negative score suggests the points may have been assigned to the wrong clusters.
    """)
    st.metric(label=f"Silhouette Score (k={num_clusters})", value=f"{sil_score:.2f}")  # Display the silhouette score    


st.markdown("---")
st.caption("Built using Streamlit and Visual Studio Code| Unsupervised ML Demo App")