#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 16:26:07 2025

@author: aaishakeshari
"""


import pandas as pd
import streamlit as st
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Download required resources
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

positive_examples = [
    "Thank you so much for the detailed portfolio review — very impressed with the service!",
    "Client was thrilled with the returns on their DPM strategy. Wants to allocate more next quarter.",
    "Really appreciate the quick turnaround on the mortgage solution — excellent work!",
    "Had a great meeting today. Client said they feel very well looked after.",
    "Client shared that the new onboarding experience was smooth and seamless.",
    "Thanks again for the CIO note. Very insightful — shared it with my partner too.",
    "Client mentioned how responsive the RM has been lately — very satisfied.",
    "She said she’s extremely happy with how the team handled the FX transfer.",
    "Wonderful call. Client wants to explore even more investment options with us.",
    "Positive feedback from client about how proactive we’ve been in communicating market updates."
]

negative_examples = [
    "He expressed disappointment with the recent fund performance and lack of explanation.",
    "Unhappy with the fees and asked for a formal review of the mandate.",
    "She said she’s been waiting too long for the RM to get back about the portfolio rebalancing.",
    "Said he doesn’t feel heard — raised concerns about lack of personalization.",
]



# Streamlit UI
st.set_page_config(page_title="Client NLP Insight Tool", layout="centered")
st.title("Demo: NLP Insight from Client Communication")

st.markdown("Paste a call report or email below to extract sentiment, commitments, and key topics.")

user_input = st.text_area("✍Enter text", height=200)

if st.button("Analyze") and user_input.strip():
    sentiment_score = sia.polarity_scores(user_input)['compound']
    sentiment_label = (
        'Positive' if sentiment_score > 0.2 else
        'Negative' if sentiment_score < -0.2 else
        'Neutral'
    )

    st.subheader("🔎 Analysis Results")
    st.markdown(f"**Sentiment Score**: {sentiment_score:.3f}")
    st.markdown(f"**Sentiment Label**: `{sentiment_label}`")

df_sentiment_examples = pd.DataFrame({
    "Text": positive_examples + negative_examples,
    "ExpectedSentiment": ["Positive"] * 10 + ["Negative"] * 4
})

st.subheader("Clearly Positive vs Negative Examples")
st.dataframe(df_sentiment_examples, use_container_width=True)


