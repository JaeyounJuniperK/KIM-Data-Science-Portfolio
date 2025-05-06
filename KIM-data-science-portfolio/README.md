# KIM-Data-Science-Portfolio
This repository is for data science projects required for the Introduction to Data Science course and will be updated accordingly. 

## 1. Portfolio Update #1: [**Interactive Palmers Penguins Dataset **](https://github.com/JaeyounJuniperK/KIM-Data-Science-Portfolio/tree/main/KIM-data-science-portfolio/basic_streamlit_app) 
This streamlit app displays the Palmers Penguins dataset and allows filtering the data interactively. User can filter data by species, island, and bill length. 

In this project, I developed an interactive web application using Streamlit to explore the Palmers Penguins Dataset. The dataset contains information on three species of penguins (Adélie, Chinstrap, and Gentoo) and includes measurements such as bill length, bill depth, flipper length, and body mass. The goal of this app is to allow users to explore and filter the dataset interactively to gain insights into the relationships between the different variables and species. 

The app also allows users to observe patterns in the dataset. For example, users can compare how the bill length differs across species, or see how flipper length and body mass correlate for penguins across different islands. It offers a simple yet powerful way to gain insights into the power of data visualizations and organization.

This app is a great introduction to my overall portfolio because it showcases my ability to clean, manipulate, and visualize data, as well as my understanding of basic data science techniques like exploratory data analysis. The interactive nature of the app highlights my ability to create accessible tools for users, making complex data easier to understand and engage with. This is an essential skill for communicating data insights effectively and bridging the gap between data science and non-technical users.

## 2. Portfolio Update #2: Beijing Olympic Medalist Tidy Data
The notebook aims to demonstrate how to import, tidy, and visualize data from the [**2008 Summer Olympic Games**](https://edjnet.github.io/OlympicsGoNUTS/2008/). It walks through the process of reshaping the original dataset (which is in a wide format) into a long format that adheres to the principles of tidy data. This transformation makes the data easier to filter, group, and analyze using the Pandas library. 

**What is Tidy Data**
Tidy data is a way of organizing data that makes it easier to analyze and visualize. It follows a specific structure that is consistent and predictable, making data manipulation more straightforward. Tidy data is important because it makes data analysis and visualization more efficient and intuitive. Reference: [**Tidy Data**](https://vita.had.co.nz/papers/tidy-data.pdf)

*The key principles of tidy data include the following:*
- Each variable forms a column: Every column in the dataset represents a specific variable or attribute.
- Each observation forms a row: Every row contains a single observation or record.
- Each type of observational unit forms a cell: Different types of data (e.g., athletes, events, results) are stored in separate cells.

In short, tidy data makes the data analysis process more organized, reliable, and efficient!

## 3. Portfolio Update #3: Supervised Machine Learning Streamlit App
This project is an Supervised Machine Learning built with Streamlit, designed to make machine learning more accessible and engaging. It allows users to experiment with various models, datasets, and hyperparameters in a no-code environment. The goal is to provide a simple and visual way to understand how different algorithms behave and how performance varies based on model selection and configuration. 

Users can either upload their own CSV datasets or choose from built-in sample datasets like the Iris dataset (for classification) or the Diabetes dataset (used here in place of the deprecated Boston housing dataset, for regression). After loading a dataset, users can select the target variable and input features, specify whether the task is classification or regression, and then choose a model to apply. 

The app includes four machine learning models: Linear Regression, Logistic Regression, Decision Tree, and K-Nearest Neighbors. Depending on the model selected, users can tune hyperparameters such as max_depth for decision trees and n_neighbors for KNN using intuitive sliders in the sidebar. Once configured, the app splits the data, trains the model, and outputs performance metrics such as accuracy, precision, R² score, and visualizations like scatter plots or ROC curves.

Building this app deepened my understanding of the end-to-end machine learning workflow, from data preprocessing and model selection to training, evaluation, and visualization. It also helped me gain practical experience integrating machine learning with interactive UI development using Streamlit, making technical concepts more accessible to non-coders.

This project is a valuable addition to my portfolio because it demonstrates my ability to combine data science with interactive web development in Python. It highlights not only my knowledge of machine learning but also my ability to create polished, user-focused tools and deploy them for public use—skills that are essential for real-world data applications and product-oriented roles.

[**Supervised Machine Learning**](https://kim-data-science-portfolio-cmbsmxkiarxtgwmiiagpqe.streamlit.app/)

## 4. Portfolio Update #4: Unsupervised Machine Learning Streamlit App
This Streamlit web application serves as an interactive platform for exploring unsupervised machine learning techniques, specifically clustering using the K-Means algorithm. Users can upload their own datasets or select from built-in samples like the Iris and Wine datasets. The app guides users through selecting numeric features, applying normalization, choosing the number of clusters, and visualizing the results through PCA-reduced scatter plots, Elbow plots, and silhouette scores. This project deepens my practical understanding of unsupervised learning by putting theory into action—transforming data through scaling, reducing dimensions with PCA, and interpreting clustering performance. It reinforces core ML concepts such as feature selection, model evaluation, and the importance of preprocessing for algorithm effectiveness.

This app complements my overall portfolio by demonstrating not only my ability to apply unsupervised learning algorithms but also my skills in interactive Python development using Streamlit. It showcases the deployment of machine learning models in a way that allows users to actively engage with the data and understand the effects of their decisions on model outcomes. By adding this unsupervised learning project, I am expanding the breadth of my machine learning skills while continuing to develop user-friendly, interactive web tools that can be easily accessed by non-experts.

[**Unsupervised Machine Learning**](https://kim-data-science-portfoliomlunsupervisedapp-luyjg7.streamlit.app/ )

