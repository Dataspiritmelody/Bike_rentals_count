import pickle
import streamlit as st
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model (you can replace this with your model)
with open('bike_rental_regressor.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the pre-fitted scaler (if used during training)
with open('scaler.pickle', 'rb') as f:
    scaler = pickle.load(f)

def main():
    st.title('Bike Rental Predictor')

    # Collect input features from the user
    hour = st.number_input('Hour', min_value=0, max_value=23)
    temperature = st.number_input('Temperature (°C)')
    humidity = st.number_input('Humidity (%)')
    wind_speed = st.number_input('Wind Speed (m/s)')
    visibility = st.number_input('Visibility (10m)')
    dew_point_temp = st.number_input('Dew Point Temperature (°C)')
    solar_radiation = st.number_input('Solar Radiation (MJ/m2)')
    rainfall = st.number_input('Rainfall (mm)')
    snowfall = st.number_input('Snowfall (cm)')
    season = st.selectbox('Season', ['Spring', 'Summer', 'Fall', 'Winter'])
    holiday = st.selectbox('Holiday', ['No', 'Yes'])
    functioning_day = st.selectbox('Functioning Day', ['No', 'Yes'])

    # Convert categorical inputs to numeric (e.g., via encoding)
    season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    holiday_map = {'No': 0, 'Yes': 1}
    functioning_map = {'No': 0, 'Yes': 1}

    season_val = season_map[season]
    holiday_val = holiday_map[holiday]
    functioning_val = functioning_map[functioning_day]

    # Create input array for prediction
    features = np.array([[hour, temperature, humidity, wind_speed, visibility, dew_point_temp, 
                          solar_radiation, rainfall, snowfall, season_val, holiday_val, functioning_val]])

    # Standardize features if scaler was used during model training
    features_std = scaler.transform(features)

    if st.button('Predict'):
        # Make prediction using the model
        prediction = model.predict(features_std)
        st.success(f'Predicted Bike Rentals: {int(prediction[0])}')

if __name__ == '__main__':
    main()
