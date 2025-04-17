# Supervised Machine Learning (Streamlit App)
## Project Overview: 
This app provides a hands-on way to experiment with machine learning models using your own data or sample datasets. Whether you're learning the basics or testing models quickly, this tool lets you visualize results, tune hyperparameters, and compare performance in real-time.

## Instructions: 
To run the app locally, follow these steps:
Clone the project repository from GitHub and navigate into the project folder on your computer.
Install the required Python libraries. You can do this using a requirements.txt file or by installing each library manually.
Run the Streamlit application using the Streamlit CLI. 
Once started, it will open in your default web browser.
Interact with the app by uploading your dataset or selecting a sample dataset, choosing features and target variables, configuring a model, and reviewing performance metrics.
If you prefer not to run the app locally, you can access the deployed version online through the provided Streamlit Cloud link.

## App Features: 
This app allows users to explore both regression and classification tasks using popular machine learning models. Users can interactively choose models, select features, and fine-tune hyperparameters from the sidebar. 

## Models Included: 
- Linear Regression: A basic regression model used for predicting continuous numerical values. No hyperparameters are exposed for tuning, making it simple for quick experimentation.
- Logistic Regression: A classification model suitable for binary or multi-class classification problems. Automatically handles feature scaling and uses default hyperparameters with increased maximum iterations (max_iter=1000) for stability.
- Decision Tree: A flexible model for both regression and classification tasks.
- K-Nearest Neighbors: A distance-based model that makes predictions based on the closest data points.
Hyperparameters are adjusted in real-time using interactive widgets in the sidebar:
- Sliders are used for numerical hyperparameters like max_depth and n_neighbors.
- Users can instantly see how changes in hyperparameters affect model performance, enabling hands-on experimentation and model comparison. 
This interactive setup encourages a better understanding of how different models behave and how hyperparameter tuning can influence results.

## References: 
- Streamlit Documentation
Comprehensive guide on building interactive web apps with Streamlit
https://docs.streamlit.io
- Scikit-learn Documentation
Official documentation for machine learning models, metrics, and utilities
https://scikit-learn.org/stable/user_guide.html
- Pandas Documentation
Reference for data manipulation and analysis using DataFrames
https://pandas.pydata.org/docs/
- Matplotlib Documentation
Guide to creating static, animated, and interactive visualizations in Python
https://matplotlib.org/stable/index.html
- Seaborn API Reference
High-level interface for drawing attractive statistical graphics
https://seaborn.pydata.org/api.html