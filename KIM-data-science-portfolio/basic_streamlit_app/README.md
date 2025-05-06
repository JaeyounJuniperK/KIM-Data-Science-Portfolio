# Describing the App

Portfolio Update #1: This app displays the Palmers Penguins dataset and allows filtering the data interactively. 
The data from the CSV file is displayed to its user. 
Users can filter data by species, island, and bill length.
Based on these filters, the data will be filtered and only values of relevance will be displayed in the filtered data below. 
Additionally, the number of values remaining after the filters are applied are displayed. 

# Palmers Penguins Dataset Interactive Explorer

This is a Streamlit app that allows users to explore the **Palmers Penguins Dataset** interactively. The app provides functionality to filter and view penguin data based on various criteria such as species, island, and bill length. It's designed to help users better understand the dataset through an intuitive, no-code interface.

## Overview

The **Palmers Penguins Dataset** contains information about three species of penguins—Adélie, Chinstrap, and Gentoo. The dataset includes measurements such as bill length, bill depth, flipper length, body mass, and the island where the penguins were found. This app allows users to explore the dataset and apply interactive filters to gain insights.

## Features

- **Species Filter**: Select a specific penguin species (Adélie, Chinstrap, or Gentoo) to focus on.
- **Island Filter**: Filter the data based on the island where the penguins were observed (Biscoe, Dream, or Torgersen).
- **Bill Length Filter**: Adjust a slider to select a range of bill lengths in millimeters, allowing for a more focused analysis of the data.
- **Data Display**: View the filtered dataset and get a detailed, tabular representation of the penguin data based on the applied filters.
- **Interactive UI**: The app is powered by Streamlit, providing a simple and intuitive interface that requires no coding experience.

## Installation and Setup
To install locally, follow the guidelines below: 
- Navigate to the project directory: cd repository-name
- Install the required libraries using pip: pipreqs
- Initiate the Streamlit App: streamlit run app.py
  
Once the app starts, Streamlit will provide a URL in the terminal. Open that URL in your browser to interact with the app.
