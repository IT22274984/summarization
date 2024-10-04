import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from summarize import NewsSummarization, KeywordExtraction, TopicModeling

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Load transformer model for abstractive summarization
hub_model_id = "shivaniNK8/t5-small-finetuned-cnn-news"
summarizer = pipeline("summarization", model=hub_model_id, use_fast=False)

# Set page configuration for better responsiveness
st.set_page_config(layout="wide")

# Inject custom CSS for background and theme color changes
st.markdown(
    """
    <style>
    /* Change overall background color */
    .main {
        background-color: #c8cce0;
    }
    
    /* Change sidebar background color */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
    
    /* Style for buttons */
    .stButton>button {
        background-color: #4B8BBE;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }

    /* Text area style */
    .stTextArea>textarea {
        background-color: #FFFFFF;
        color: #333333;
        border: 1px solid #D3D3D3;
    }

    /* Headings color */
    h1, h2, h3 {
        color: #4B8BBE;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Define layout: left column (main content) and right column (user input options)
left_col, right_col = st.columns([2.5, 1])

# Left side (main content)
with left_col:
    # Page header
    st.write("""# HIGHLIGHTS! \n ### A News Summarizer""")
    st.write("Provide a news article and get a summary within seconds!")

    # Image for aesthetic purposes with specified width
    image = Image.open('C:\\Users\\shant\\OneDrive\\Desktop\\summerize\\News-Article-Text-Summarizer-Transformer-master\\StreamlitApp\\newspaper3.jpeg')
    image = image.resize((500, 350))  # Resize image to width=450 and height=250
    st.image(image)  # Only width is specified here

    # Input article from the user
    article = st.text_area(label="Enter the article you want to summarize", height=250, placeholder="Enter Article Body Here")

    # Sentiment Analysis class
    class SentimentAnalysis:
        def __init__(self):
            self.sid = SentimentIntensityAnalyzer()

        def analyze_sentiment(self, text):
            scores = self.sid.polarity_scores(text)
            return scores

    # Instantiate classes for summarization, keyword extraction, and topic modeling
    news_summarizer = NewsSummarization()
    keyword_extractor = KeywordExtraction()
    topic_modeler = TopicModeling()

    # Perform summarization and analysis if the button is clicked
    if st.session_state.get('submitted', False):
        # Perform extractive summarization
        st.write("## Extractive Summary")
        ex_summary = news_summarizer.extractive_summary(article, num_sentences=st.session_state.max_sentences)
        st.write(ex_summary)

        # Perform abstractive summarization using transformer model
        summary = summarizer(article, max_length=st.session_state.max_words, min_length=st.session_state.min_words, do_sample=False, clean_up_tokenization_spaces=True)
        abs_summary = summary[0]['summary_text']
        st.write("## Abstractive Summary")
        st.write(abs_summary)

        # Perform sentiment analysis
        sentiment_analyzer = SentimentAnalysis()
        sentiment_scores = sentiment_analyzer.analyze_sentiment(article)
        st.write("## Sentiment Analysis")
        st.write(sentiment_scores)

        # Perform keyword extraction
        st.write("## Keyword Extraction")
        keywords = keyword_extractor.extract_keywords(article, top_n=10)
        st.write(keywords)

        # Perform topic modeling
        st.write("## Topic Modeling")
        topics = topic_modeler.extract_topics(article, num_topics=5)
        st.write(topics)

# Right side (user input options)
with right_col:
    st.markdown("### Summary Options")
    st.write("Set parameters for summarization:")

    # Max sentences and words inputs in a more professional way
    max_sentences = st.number_input('Summary Length (Extractive)', min_value=1, max_value=10, value=3, key="max_sentences")
    max_words = st.number_input('Max words (Abstractive)', min_value=50, max_value=500, step=10, value=200, key="max_words")
    min_words = st.number_input('Min words (Abstractive)', min_value=10, max_value=450, step=10, value=100, key="min_words")

    # Submit button
    submit_button = st.button("Summarize!", key="submitted")

    # Sidebar information on summarization methods (now placed under the button)
    with st.expander("More About Summarization"):
        st.markdown(""" 
        *Extractive Summarization*: Identifies and selects important sentences from the article.
        
        *Abstractive Summarization*: Generates a summary by interpreting the context and producing new sentences.
        """)