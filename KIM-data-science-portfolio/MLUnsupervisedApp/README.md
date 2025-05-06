**📌 Project Overview**
The Unsupervised Machine Learning Explorer is an interactive Streamlit web application designed to help users explore and understand clustering techniques—particularly K-Means—in a hands-on, visual way. The goal of the project is to make unsupervised learning approachable and intuitive for learners, educators, and data enthusiasts.

With this app, users can:
- Upload their own dataset or use built-in sample datasets (i.e. Iris and Wine).
- Select which numeric features to include in clustering.
- Apply normalization using StandardScaler.
- Choose the number of clusters (k) for K-Means.
- Visualize clusters in 2D using PCA.
- Evaluate clustering quality using the Elbow Method and Silhouette Score.

The app requires no coding knowledge and provides immediate visual feedback, making it a great educational tool for understanding the principles of clustering.

**🛠️ Instructions**
To run the Unsupervised ML Explorer locally on your machine, follow these steps:
- Clone the project repository from GitHub and navigate into the project folder on your computer.
- Install the required Python libraries. You can do this using a requirements.txt file or by installing each library manually.
- Run the Streamlit application using the Streamlit CLI. 

If you prefer not to run the app locally, you can access the deployed version online through the provided Streamlit Cloud link.

Once started, it will open in your default web browser.
Interact with the app by uploading your dataset or selecting a sample dataset, choosing features and target variables, configuring a model, and reviewing performance metrics.

**⚙️ App Features** 
The Unsupervised ML Explorer is designed to be intuitive and interactive, offering an end-to-end experience for exploring clustering algorithms. Here's how you can interact with the app: 
1. Upload or Select a Dataset: Upload your own CSV file containing numeric features you want to cluster. Or, choose from two built-in sample datasets. (Iris Dataset: Includes measurements of iris flowers (sepal and petal dimensions) with known species labels. Wine Dataset: Contains chemical properties of wines from three grape varieties.) 

Tip: Your CSV should include a header row and numeric columns for feature selection.

2. Select Features for Clustering: Choose from available numeric columns to use as input features. You must select at least two features for clustering. You can optionally apply feature normalization using StandardScaler to ensure that all features contribute equally. 

Note: Avoid selecting features that are too highly correlated or irrelevant, as they can distort clustering results.

3. Configure the K-Means Model: Use the sidebar slider to select the number of clusters (k) you want the K-Means algorithm to find (between 2 and 10). The app automatically fits a KMeans model and assigns each data point to a cluster.

4. Visualize Results & Evaluate Performance: Explore the following output tabs: 
 - Cluster Plot: View a 2D projection of your clustered data using PCA (Principal Component Analysis). Each point is colored by its assigned cluster. 
 - Elbow Plot: Helps identify the optimal number of clusters (k) by visualizing inertia (within-cluster sum of squares). 
 - Silhouette Score: A score from -1 to 1 indicating how well-defined your clusters are. Higher is better.

**📚 References** 
- [Streamlit Documentation](https://docs.streamlit.io/) – For building the interactive web app.
- [Pandas Documentation](https://pandas.pydata.org/docs/) – For handling and manipulating datasets.
- [K-Means Clustering — A Hands-On scikit-learn Tutorial](https://realpython.com/k-means-clustering-python/) – A practical guide to understanding KMeans with Python.
- [The Elbow Method Explained](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/) – Explanation of how to find the optimal number of clusters. 
- [Silhouette Score Explained](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) – Official API reference and interpretation guide.
