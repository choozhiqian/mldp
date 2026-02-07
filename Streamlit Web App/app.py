import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="📈 Instagram Followers Gain Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../Model Development/followers_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# App Title & Instructions
# -----------------------------
st.title("📈 Instagram Followers Gain Predictor")
st.markdown("""
Enter the metrics of your Instagram post below.  
Click **Predict** to see an estimate of followers gained.
""")

# -----------------------------
# Input Form
# -----------------------------
with st.form(key="input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        reach = st.number_input("Reach", min_value=0, step=1)
        likes = st.number_input("Likes", min_value=0, step=1)
        
    with col2:
        comments = st.number_input("Comments", min_value=0, step=1)
        saves = st.number_input("Saves", min_value=0, step=1)
    
    submit_button = st.form_submit_button(label="Predict")
    clear_button = st.form_submit_button(label="Clear Inputs")

# -----------------------------
# Clear Inputs
# -----------------------------
if clear_button:
    st.experimental_rerun()

# -----------------------------
# Prediction Logic
# -----------------------------
if submit_button:
    # Input validation
    if reach == 0:
        st.error("❌ Reach cannot be zero. Please enter a positive number.")
    else:
        total_engagement = likes + comments + saves
        engagement_per_reach = total_engagement / reach  # safe division
        
        input_df = pd.DataFrame({
            'reach': [reach],
            'likes': [likes],
            'comments': [comments],
            'saves': [saves],
            'total_engagement': [total_engagement],
            'engagement_per_reach': [engagement_per_reach]
        })
        
        prediction = model.predict(input_df)
        st.success(f"✅ Estimated Followers Gained: {int(prediction[0])}")

        # Engagement Breakdown Chart
        st.subheader("Engagement Breakdown")
        labels = ['Likes', 'Comments', 'Saves']
        values = [likes, comments, saves]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures pie is circular
        st.pyplot(fig)

# -----------------------------
# Footer / Info
# -----------------------------
st.markdown("""
---
**Note:** This predictor is based on a trained model and provides an estimate.  
It does not guarantee exact follower gains but highlights trends in engagement.
""")
