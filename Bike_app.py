import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the model and scaler
with open('best_bike_rental_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Function to make predictions
def predict_bike_rental_category(temperature, humidity, month, solar_radiation, hour):
    # Create DataFrame for new data
    new_data = pd.DataFrame({
        'Temperature(°C)': [temperature],
        'Humidity(%)': [humidity],
        'Month': [month],
        'Solar Radiation (MJ/m2)': [solar_radiation],
        'Hour': [hour]
    })
    
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    
    # Map prediction to label (low, medium, high)
    label_encoder_category = LabelEncoder()
    label_encoder_category.fit(['low', 'medium', 'high'])  # Ensure these match your actual labels
    
    return label_encoder_category.inverse_transform(prediction)[0]

# Streamlit UI
st.title("Bike Rental Category Prediction")
st.write("Enter the values to predict the bike rental category (Low, Medium, High)")

# Create input fields
temperature = st.slider("Temperature (°C)", min_value=-30.0, max_value=50.0, value=15.0, step=0.1)
humidity = st.slider("Humidity (%)", min_value=0, max_value=100, value=50)
month = st.selectbox("Month", options=list(range(1, 13)), index=5)
solar_radiation = st.slider("Solar Radiation (MJ/m2)", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
hour = st.selectbox("Hour of the Day", options=list(range(0, 24)), index=12)

# Predict button
if st.button("Predict Bike Rental Category"):
    prediction = predict_bike_rental_category(temperature, humidity, month, solar_radiation, hour)
    
    # Display the result
    st.success(f"The predicted bike rental category is: **{prediction.capitalize()}**")
