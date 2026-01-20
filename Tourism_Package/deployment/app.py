import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rahulsuren12/tourism-package-model", filename="best_tourism_package_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction")
st.write("""
This app predicts whether a customer is likely to purchase a tourism package.
""")

# -------------------------
# User Inputs
# -------------------------
age = st.number_input("Age", min_value=18, max_value=80, value=35)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=25000)
number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=2)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)

gender = st.selectbox("Gender", ["Male", "Female"])
occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Small Business", "Large Business"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
city_tier = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe"])


# -------------------------
# Assemble input
# -------------------------
input_df = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": monthly_income,
    "NumberOfTrips": number_of_trips,
    "DurationOfPitch": duration_of_pitch,
    "PreferredPropertyStar": preferred_star,
    "NumberOfPersonVisiting": num_persons,
    "Gender": gender,
    "Occupation": occupation,
    "MaritalStatus": marital_status,
    "CityTier": city_tier,
    "ProductPitched": product_pitched
}])


# -------------------------
# Prediction
# -------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Customer is likely to purchase the package (Probability: {probability:.2f})")
    else:
        st.warning(f"Customer is unlikely to purchase the package (Probability: {probability:.2f})")
