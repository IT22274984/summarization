import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from summarize import NewsSummarization

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load transformer model for abstractive summarization
hub_model_id = "shivaniNK8/t5-small-finetuned-cnn-news"
summarizer = pipeline("summarization", model=hub_model_id)

# Page header
st.write("""# HIGHLIGHTS! \n ### A News Summarizer""")
st.write("Provide a news article and get a summary within seconds!")

# Image for aesthetic purposes
image = Image.open('newspaper.jpeg')
st.image(image)

# Sidebar for user input options
st.sidebar.header('Select summary parameters')
with st.sidebar.form("input_form"):
    st.write('Select summary length for extractive summary:')
    max_sentences = st.slider('Summary Length (Extractive)', 1, 10, step=1, value=3)
    
    st.write('Select word limits for abstractive summary:')
    max_words = st.slider('Max words (Abstractive)', 50, 500, step=10, value=200)
    min_words = st.slider('Min words (Abstractive)', 10, 450, step=10, value=100)
    
    submit_button = st.form_submit_button("Summarize!")

# Input article from the user
article = st.text_area(label="Enter the article you want to summarize", height=300, value="Enter Article Body Here")

# Instantiate NewsSummarization class for extractive summaries
news_summarizer = NewsSummarization()

# Sentiment Analysis class
class SentimentAnalysis:
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text):
        scores = self.sid.polarity_scores(text)
        return scores

if submit_button:
    # Perform extractive summarization
    st.write("## Extractive Summary")
    ex_summary = news_summarizer.extractive_summary(article, num_sentences=max_sentences)
    st.write(ex_summary)

    # Perform abstractive summarization using transformer model
    summary = summarizer(article, max_length=max_words, min_length=min_words, do_sample=False, clean_up_tokenization_spaces=True)
    abs_summary = summary[0]['summary_text']
    st.write("## Abstractive Summary")
    st.write(abs_summary)

    # Perform sentiment analysis
    sentiment_analyzer = SentimentAnalysis()
    sentiment_scores = sentiment_analyzer.analyze_sentiment(article)
    st.write("## Sentiment Analysis")
    st.write(sentiment_scores)

# Sidebar information on summarization methods
with st.sidebar.expander("More About Summarization"):
    st.markdown(""" 
    **Extractive Summarization**: Identifies and selects important sentences from the article.
    
    **Abstractive Summarization**: Generates a summary by interpreting the context and producing new sentences.
    """)

# Example dataset to show results (optional)
# df = pd.read_csv('dataset.csv')
# st.write(df)
