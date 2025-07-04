import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Skin Type Predictor", layout="centered")

st.title("🧴 Skin Type Prediction App")
st.markdown("Answer the questions below to find out your skin type and get helpful skincare tips!")

# Questions and options (must match training data)
after_wash_feel = st.selectbox("1. How does your skin feel 2 hours after washing your face?", 
                                ["Tight", "Normal", "Slightly oily", "Very oily"])

flakiness = st.selectbox("2. Do you experience flakiness or dry patches?", 
                         ["Yes", "No"])

acne = st.selectbox("3. Do you get acne or pimples?", 
                    ["Yes", "No"])

shine = st.selectbox("4. How shiny does your face look by midday?", 
                     ["Never", "Sometimes", "Always"])

sensitivity = st.selectbox("5. How sensitive is your skin to new products?", 
                           ["Low", "Moderate", "High"])

# Input collection
user_inputs = {
    "after_wash_feel": after_wash_feel,
    "flakiness": flakiness,
    "acne": acne,
    "shine": shine,
    "sensitivity": sensitivity
}

if st.button("Predict Skin Type"):
    try:
        # Encode input using stored label encoders
        encoded_input = []
        for col, val in user_inputs.items():
            le = label_encoders[col]
            encoded_val = le.transform([val])[0]
            encoded_input.append(encoded_val)

        # Make prediction
        prediction = model.predict([encoded_input])
        skin_type = target_encoder.inverse_transform(prediction)[0]

        st.success(f"🧬 Your predicted skin type is: **{skin_type}**")

        # Optional: Tips based on skin type
        tips = {
            "Dry": "💧 Use a hydrating moisturizer and avoid harsh cleansers.",
            "Oily": "🧼 Use oil-free products and blotting paper to reduce shine.",
            "Normal": "✅ Maintain your routine and stay hydrated!",
            "Combination": "🌗 Use different products for T-zone and cheeks.",
            "Sensitive": "⚠️ Use fragrance-free products and patch-test new ones."
        }

        st.info(tips.get(skin_type, "Take care of your skin!"))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
