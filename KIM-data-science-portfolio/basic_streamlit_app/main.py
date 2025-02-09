import streamlit as st
import pandas as pd

st.title('Palmers Penguins Dataset')
st.write("""
This app displays the Palmers Penguins dataset and allows filtering the data interactively.
User can filter data by species, island, and bill length.
""")

df = pd.read_csv("data/penguins.csv")
st.write("Here's the dataset loaded from a CSV file:")
st.dataframe(df)

species_filter = st.selectbox('Select species', df['species'].unique())
island_filter = st.selectbox('Select island', options=df['island'].unique())
bill_length_filter = st.slider('Select bill length (mm)', min_value=float(df['bill_length_mm'].min()), 
                               max_value=float(df['bill_length_mm'].max()), 
                               value=(float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max())))

filtered_penguins = df[
    (df['species'] == species_filter) &
    (df['island'] == island_filter) &
    (df['bill_length_mm'] >= bill_length_filter[0]) &
    (df['bill_length_mm'] <= bill_length_filter[1])
]

st.write(f"Displaying {filtered_penguins.shape[0]} records based on  filters.")
st.dataframe(filtered_penguins)