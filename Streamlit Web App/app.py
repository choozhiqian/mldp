import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("../Model Development/followers_model.pkl")

st.set_page_config(page_title="Followers Predictor")

st.title("📈 Instagram Followers Gain Predictor")

reach = st.number_input("Reach", min_value=0)
likes = st.number_input("Likes", min_value=0)
comments = st.number_input("Comments", min_value=0)
saves = st.number_input("Saves", min_value=0)

if st.button("Predict"):

    total_engagement = likes + comments + saves
    engagement_per_reach = total_engagement / (reach + 1)

    input_df = pd.DataFrame({
        'reach': [reach],
        'likes': [likes],
        'comments': [comments],
        'saves': [saves],
        'total_engagement': [total_engagement],
        'engagement_per_reach': [engagement_per_reach]
    })

    prediction = model.predict(input_df)

    st.success(f"Estimated Followers Gained: {int(prediction[0])}")
