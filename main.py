import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained SVM classifier from a saved file
classifier = pickle.load(open('trained_model.sav', 'rb'))

# Load the diabetes dataset to fit the StandardScaler
diabetes_dataset = pd.read_csv('/content/diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)  # All features except the target variable
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler with the original dataset features

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Title and sidebar
st.title('Diabetes Prediction App')

st.sidebar.title("Input Here")

# Input fields for required data
pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=200, value=80)
bmi = st.sidebar.number_input('BMI', min_value=0, max_value=100, value=25)
diabetes_pedigree_function = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, step=0.01, value=0.5)
age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=30)

# Prediction button
if st.sidebar.button('Predict Diabetes'):
    # Create a numpy array from the input data
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Standardize the input data
    standardized_input_data = scaler.transform(input_data_as_numpy_array)

    # Make a prediction with the loaded classifier
    prediction = classifier.predict(standardized_input_data)

    # Display the result
    if prediction[0] == 0:
        st.success('THE PERSON IS NON DIABETIC.')
    else:
        st.warning('THE PERSON IS DIABETIC.')
