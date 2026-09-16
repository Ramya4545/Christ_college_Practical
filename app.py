import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("Christ_college_accident_prediction_model.pkl")


# Page configuration
st.set_page_config(
    page_title="Traffic Accident Risk Prediction",
    page_icon="🚦",
    layout="centered"
)

# Title
st.title("🚦 Smart Traffic Accident Risk Prediction")
st.write("Enter the traffic details to predict accident risk.")

# User inputs
speed = st.number_input(
    "Vehicle Speed (km/h)",
    min_value=0,
    max_value=200,
    value=50
)

weather = st.selectbox(
    "Weather",
    ["Clear", "Rain", "Fog", "Cloudy"]
)

traffic = st.selectbox(
    "Traffic Condition",
    ["Low", "Medium", "High"]
)

road_type = st.selectbox(
    "Road Type",
    ["Highway", "City Road", "Village Road", "Intersection"]
)

time = st.selectbox(
    "Time",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# Prediction button
if st.button("Predict Accident Risk"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Speed": [speed],
        "Weather": [weather],
        "Traffic": [traffic],
        "Road_Type": [road_type],
        "Time": [time]
    })

    # IMPORTANT:
    # Apply the same encoding/preprocessing used during training.
    input_data = pd.get_dummies(input_data)

    # Make sure input columns match training columns
    if hasattr(model, "feature_names_in_"):
        input_data = input_data.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    # Prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"Predicted Accident Risk: {prediction}")
