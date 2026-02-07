import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO
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
# Load Trained Model from Google Drive
# -----------------------------
@st.cache_data(show_spinner=True)
def load_model_from_drive(drive_url):
    """
    Load a joblib model from a Google Drive shareable link.
    """
    # Convert Google Drive share link to direct download
    file_id = drive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    response = requests.get(download_url)
    response.raise_for_status()
    
    model = joblib.load(BytesIO(response.content))
    return model

MODEL_URL = "https://drive.google.com/file/d/1EDx0nofrfgFAB4fDTjOdsTUxOHPUnBw0/view?usp=sharing"
model = load_model_from_drive(MODEL_URL)

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
        reach = st.number_input("Reach", min_value=0, step=1, key="reach")
        likes = st.number_input("Likes", min_value=0, step=1, key="likes")
        
    with col2:
        comments = st.number_input("Comments", min_value=0, step=1, key="comments")
        saves = st.number_input("Saves", min_value=0, step=1, key="saves")
    
    submit_button = st.form_submit_button(label="Predict")
    clear_button = st.form_submit_button(label="Clear Inputs")

# -----------------------------
# Clear Inputs
# -----------------------------
if clear_button:
    st.session_state["reach"] = 0
    st.session_state["likes"] = 0
    st.session_state["comments"] = 0
    st.session_state["saves"] = 0

# -----------------------------
# Prediction Logic
# -----------------------------
if submit_button:
    if reach == 0:
        st.error("❌ Reach cannot be zero. Please enter a positive number.")
    else:
        total_engagement = likes + comments + saves
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
        st.success(f"✅ Estimated Followers Gained: {int(prediction[0])}")

        # Engagement Breakdown Chart
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
