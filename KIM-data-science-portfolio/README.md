# KIM-Data-Science-Portfolio
This repository is for data science projects required for the Introduction to Data Science course and will be updated accordingly. 

## 1. Portfolio Update #1: Interactive Palmers Penguins Dataset
This streamlit app displays the Palmers Penguins dataset and allows filtering the data interactively. User can filter data by species, island, and bill length. 

## 2. Portfolio Update #2: Beijing Olympic Medalist Tidy Data
The notebook aims to demonstrate how to import, tidy, and visualize data from the [**2008 Summer Olympic Games**](https://edjnet.github.io/OlympicsGoNUTS/2008/). It walks through the process of reshaping the original dataset (which is in a wide format) into a long format that adheres to the principles of tidy data. This transformation makes the data easier to filter, group, and analyze using the Pandas library. 

**What is Tidy Data**
Tidy data is a way of organizing data that makes it easier to analyze and visualize. It follows a specific structure that is consistent and predictable, making data manipulation more straightforward. Tidy data is important because it makes data analysis and visualization more efficient and intuitive. Reference: [**Tidy Data**](https://vita.had.co.nz/papers/tidy-data.pdf)

*The key principles of tidy data include the following:*
- Each variable forms a column: Every column in the dataset represents a specific variable or attribute.
- Each observation forms a row: Every row contains a single observation or record.
- Each type of observational unit forms a cell: Different types of data (e.g., athletes, events, results) are stored in separate cells.

In short, tidy data makes the data analysis process more organized, reliable, and efficient!

## 3. Portfolio Update #3: Supervised Machine Learning
This project is an Supervised Machine Learning built with Streamlit, designed to make machine learning more accessible and engaging. It allows users to experiment with various models, datasets, and hyperparameters in a no-code environment. The goal is to provide a simple and visual way to understand how different algorithms behave and how performance varies based on model selection and configuration. 

Users can either upload their own CSV datasets or choose from built-in sample datasets like the Iris dataset (for classification) or the Diabetes dataset (used here in place of the deprecated Boston housing dataset, for regression). After loading a dataset, users can select the target variable and input features, specify whether the task is classification or regression, and then choose a model to apply. 

The app includes four machine learning models: Linear Regression, Logistic Regression, Decision Tree, and K-Nearest Neighbors. Depending on the model selected, users can tune hyperparameters such as max_depth for decision trees and n_neighbors for KNN using intuitive sliders in the sidebar. Once configured, the app splits the data, trains the model, and outputs performance metrics such as accuracy, precision, R² score, and visualizations like scatter plots or ROC curves.

Building this app deepened my understanding of the end-to-end machine learning workflow, from data preprocessing and model selection to training, evaluation, and visualization. It also helped me gain practical experience integrating machine learning with interactive UI development using Streamlit, making technical concepts more accessible to non-coders.

This project is a valuable addition to my portfolio because it demonstrates my ability to combine data science with interactive web development in Python. It highlights not only my knowledge of machine learning but also my ability to create polished, user-focused tools and deploy them for public use—skills that are essential for real-world data applications and product-oriented roles.
[**Supervised Machine Learning**](https://kim-data-science-portfolio-cmbsmxkiarxtgwmiiagpqe.streamlit.app/)
