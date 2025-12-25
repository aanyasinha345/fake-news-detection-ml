import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.set_page_config(page_title="Fake News Detection")
st.title("📰 Fake News Detection Interface")

news = st.text_area("Enter News Text")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some news text")
    else:
        data = tfidf.transform([news])
        prediction = model.predict(data)

        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
