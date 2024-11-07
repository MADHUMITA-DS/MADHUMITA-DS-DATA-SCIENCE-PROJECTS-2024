import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the trained model and label encoder
model = load_model("fertilizer_model.h5")

# Load the dataset to use the label encoder
# data = pd.read_csv(r"E:\customProjects\cropYield\Fertilizer_Prediction.csv")
# data = pd.read_csv(r"/mnt/Fertilizer_Prediction.csv")
data = pd.read_csv(r"Fertilizer_Prediction.csv")

label_encoder = LabelEncoder()
label_encoder.fit(data['Fertilizer_Name'])

# Streamlit app
st.title("Fertilizer Recommendation System")

st.markdown("Enter the soil content details to get the recommended fertilizer:")

# Input form
nitrogen = st.number_input("Nitrogen Content", min_value=0, max_value=100, step=1)
potassium = st.number_input("Potassium Content", min_value=0, max_value=100, step=1)
phosphorous = st.number_input("Phosphorous Content", min_value=0, max_value=100, step=1)

if st.button("Predict Fertilizer"):
    # Prepare input for prediction
    input_data = np.array([[nitrogen, potassium, phosphorous]])
    input_data = np.expand_dims(input_data, axis=-1)

    # Make prediction
    prediction = np.argmax(model.predict(input_data), axis=-1)
    fertilizer_name = label_encoder.inverse_transform(prediction)[0]

    # Display result
    st.success(f"The recommended fertilizer is: {fertilizer_name}")
