import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("MONGO_URI")

client = MongoClient(st.secrets["MONGO_URI"], uri, server_api=ServerApi('1'))

db = client['ice_cream']

collection = db["ice_cream_pred"]

def load_model():
    with open("polynomial_reg.pkl","rb") as file:
        model = pickle.load(file) # file is going to return model and scaler
    return model

def predict_data(data_dict):
    model = load_model()
    df = pd.DataFrame([data_dict])
    prediction = model.predict(df)
    return prediction


def main():
    st.title("Ice cream Sales Prediction") # Title for the app
    st.write("Enter your data to get a prediction for ice cream sales") # Normal text
    temperature = st.number_input("Temperature (in Celsius)",min_value = -30.0 , max_value = 50.0,format="%.2f")

    if st.button("Predict Sales"):
        user_data = {"Temperature (°C)": temperature}
        prediction = predict_data(user_data)
        final_value = round(float(prediction[0]), 2)
        user_data["prediction"] = final_value
        collection.insert_one(user_data)
        st.success(f"The predicted ice cream sales is: {final_value} units")

if __name__ == "__main__":
    main()