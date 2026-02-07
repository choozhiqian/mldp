import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

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
# Initialize Session State
# -----------------------------
for key in ["reach", "likes", "comments", "saves"]:
    if key not in st.session_state:
        st.session_state[key] = 0

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
        st.session_state["reach"] = st.number_input(
            "Reach", min_value=0, step=1, value=st.session_state["reach"]
        )
        st.session_state["likes"] = st.number_input(
            "Likes", min_value=0, step=1, value=st.session_state["likes"]
        )
        
    with col2:
        st.session_state["comments"] = st.number_input(
            "Comments", min_value=0, step=1, value=st.session_state["comments"]
        )
        st.session_state["saves"] = st.number_input(
            "Saves", min_value=0, step=1, value=st.session_state["saves"]
        )
    
    submit_button = st.form_submit_button(label="Predict")
    clear_button = st.form_submit_button(label="Clear Inputs")

# -----------------------------
# Clear Inputs
# -----------------------------
if clear_button:
    for key in ["reach", "likes", "comments", "saves"]:
        st.session_state[key] = 0

# -----------------------------
# Prediction Logic
# -----------------------------
if submit_button:
    # Get values from session_state
    reach = st.session_state["reach"]
    likes = st.session_state["likes"]
    comments = st.session_state["comments"]
    saves = st.session_state["saves"]

    total_engagement = likes + comments + saves

    # -----------------------------
    # Input validation
    # -----------------------------
    if reach <= 0:
        st.error("❌ Reach must be greater than zero.")
    elif total_engagement > reach:
        st.error("❌ Total engagement cannot exceed reach.")
    elif total_engagement == 0:
        st.error("❌ Engagement cannot be zero.")
    else:
        # Safe to predict
        engagement_per_reach = total_engagement / reach
        
        input_df = pd.DataFrame({
            'reach': [reach],
            'likes': [likes],
            'comments': [comments],
            'saves': [saves],
            'total_engagement': [total_engagement],
            'engagement_per_reach': [engagement_per_reach]
        })
        
        prediction = model.predict(input_df)
        
        if pd.isna(prediction[0]):
            st.error("❌ Model produced an invalid prediction. Please check your inputs.")
        else:
            st.success(f"✅ Estimated Followers Gained: {int(prediction[0])}")

            # Engagement Breakdown Chart
            if total_engagement > 0:
                st.subheader("Engagement Breakdown")
                labels = ['Likes', 'Comments', 'Saves']
                values = [likes, comments, saves]
                fig, ax = plt.subplots()
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)


# -----------------------------
# Footer / Info
# -----------------------------
st.markdown("""
---
**Note:** This predictor is based on a trained model and provides an estimate.  
It does not guarantee exact follower gains but highlights trends in engagement.
""")
