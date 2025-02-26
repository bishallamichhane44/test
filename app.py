import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model and scaler
model = joblib.load('model.sav')
scaler = joblib.load('scaler.sav')  


crop_mapping = {
    0: 'apple', 1: 'banana', 2: 'blackgram', 3: 'chickpea', 4: 'coconut',
    5: 'coffee', 6: 'cotton', 7: 'grapes', 8: 'jute', 9: 'kidneybeans',
    10: 'lentil', 11: 'maize', 12: 'mango', 13: 'mothbeans', 14: 'mungbean',
    15: 'muskmelon', 16: 'orange', 17: 'papaya', 18: 'pigeonpeas',
    19: 'pomegranate', 20: 'rice', 21: 'watermelon'
}
crop_df = pd.DataFrame(list(crop_mapping.items()), columns=['no_label', 'label'])



# Define feature names
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Streamlit UI
st.title("Crop Plantation Suggestion System")
st.write("Enter the following parameters to get a crop suggestion:")

# Create input fields
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(
        feature.capitalize(),
        min_value=0.0,
        step=0.1,
        format="%.1f"
    )

# Prediction button
if st.button("Predict Crop"):
    try:
        # Prepare input data
        input_values = [float(input_data[feature]) for feature in feature_names]
        input_array = np.array([input_values])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]

        st.write(f"Debug: Model prediction (no_label): {prediction}")  
        
        # Get crop name
        predicted_crop = crop_df[crop_df['no_label'] == prediction]['label'].iloc[0]

        
        # Display result
        st.success(f"Recommended Crop: **{predicted_crop}**")
        
    except ValueError:
        st.error("Invalid input. Please enter numerical values for all features.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Some footer information
st.write("---")




st.write("Crop Mapping Verification:", crop_df)  # For debugging