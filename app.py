import pickle
import re
import string
import streamlit as st # type: ignore

# Load saved files
model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("Tfidf_Vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

st.title("📰 Fake News Detection System")

news = st.text_area("Enter News Text")

if st.button("Predict"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(news)
        vectorized = tfidf.transform([cleaned])
        result = model.predict(vectorized)[0]

        if result == 0:
            st.error("🟥 Fake News")
        else:
            st.success("🟩 Real News")
