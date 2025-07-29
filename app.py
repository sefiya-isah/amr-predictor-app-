
import streamlit as st
import joblib
import numpy as np

# Load the trained SVM model
model = joblib.load('amr_svm_model.pkl')

st.title("🔬 AMR Prediction App")
st.write("Predict resistance to Ciprofloxacin and Augmentin")

# User input fields
imipenem = st.number_input("Imipenem (zone in mm)", min_value=0)
gentamicin = st.number_input("Gentamicin (zone in mm)", min_value=0)
ceftazidime = st.number_input("Ceftazidime (zone in mm)", min_value=0)
organism_encoded = st.selectbox("Organism", options=[("Escherichia coli", 0), ("Staphylococcus aureus", 1), ("Klebsiella pneumoniae", 2)], format_func=lambda x: x[0])[1]

# Predict button
if st.button("Predict"):
    input_data = np.array([[imipenem, gentamicin, ceftazidime, organism_encoded]])
    prediction = model.predict(input_data)
    cip, aug = prediction[0]

    result_cip = "Resistant" if cip == 1 else "Susceptible"
    result_aug = "Resistant" if aug == 1 else "Susceptible"

    st.success(f"Ciprofloxacin: {result_cip}")
    st.success(f"Augmentin: {result_aug}")
