import pickle
import streamlit as st
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

st.set_page_config(page_title="Sentiment Analysis")
st.header("Hotel Review Sentiment")

# Load the model and vectorizer
model = pickle.load(open("nlp_model.pkl", 'rb'))
vectorizer = pickle.load(open("vector.pkl", 'rb'))

def preprocess(input_text):
    if input_text.strip() == "":
        return None
    text_vectorized = vectorizer.transform([input_text])  # Transform to 2D array
    return text_vectorized

review = st.text_input('Review')

if st.button("Show sentiment"):
    text = preprocess(review)
    if text is not None:
        result = model.predict(text)[0]
        if result == 0:
            response = "Happy and satisfied"
        else:
            response = "Sad and dissatisfied"
        st.write(response)
    else:
        st.write("Please enter a valid review.")
