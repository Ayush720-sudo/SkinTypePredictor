import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load the CSV directly
@st.cache_data
def load_data_and_train_model():
    data = pd.read_csv(r"C:\Users\HP\Downloads\extended_skin_data.csv")

    X = data.drop("skin_type", axis=1)
    y = data["skin_type"]

    label_encoders = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    return model, label_encoders, target_encoder

model, label_encoders, target_encoder = load_data_and_train_model()

st.title("🧴 Skin Type Prediction")

after_wash = st.selectbox("1. How does your skin feel 2 hours after washing your face?", 
                          ["Tight", "Normal", "Slightly oily", "Very oily"])
flaky = st.selectbox("2. Do you experience flakiness or dry patches?", 
                     ["Yes", "No"])
acne = st.selectbox("3. Do you get acne or pimples?", 
                    ["Yes", "No"])
shine = st.selectbox("4. How shiny does your face look by midday?", 
                     ["Never", "Sometimes", "Always"])
sensitivity = st.selectbox("5. How sensitive is your skin to new products?", 
                           ["Low", "Moderate", "High"])

user_inputs = {
    "after_wash_feel": after_wash,
    "flakiness": flaky,
    "acne": acne,
    "shine": shine,
    "sensitivity": sensitivity
}

if st.button("Predict"):
    try:
        encoded_input = [
            label_encoders[col].transform([val])[0]
            for col, val in user_inputs.items()
        ]
        prediction = model.predict([encoded_input])
        skin_type = target_encoder.inverse_transform(prediction)[0]

        st.success(f"🌟 Your predicted skin type is: **{skin_type}**")

    except Exception as e:
        st.error(f"Error: {e}")
