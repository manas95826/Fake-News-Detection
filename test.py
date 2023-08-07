import streamlit as st
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

port_stem = PorterStemmer()
# Load the pre-trained model pipeline
with open('pipeline_model.pkl', 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

# Set up Streamlit page layout
st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="wide"
)

# Define function for text stemming
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Streamlit app header
st.title("Fake News Detection App")
st.markdown("---")

# Text input for user to enter a news article snippet
user_input = st.text_area("Enter the news article's main snippet:", "")

# Button to initiate prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a snippet of a news article.")
    else:
        # Apply stemming to the user input
        stemmed_input = stemming(user_input)
        
        # Make prediction using the model pipeline
        prediction = model_pipeline.predict([stemmed_input])
        prediction_prob = model_pipeline.predict_proba([stemmed_input])

        # Display the prediction result
        st.markdown("---")
        if prediction[0] == 1:
            st.error("🚨 Fake News Detected!")
        else:
            st.success("✅ Not Fake")

        st.markdown("---")
        st.write("Prediction Probability (Not Fake, Fake):", prediction_prob)

# Streamlit footer
st.markdown("---")
st.write("Note: This is a simple demo for Fake news detection.")
