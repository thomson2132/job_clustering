import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import normalize
from scripts.scraper import scrape_karkidi_jobs

# Load model and vectorizer with caching
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return kmeans_model, vectorizer

# Classify jobs using loaded model
def classify_new_jobs(df, model, vectorizer):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    X = vectorizer.transform(df["Skills"])
    X = normalize(X)
    df["Cluster"] = model.predict(X)
    return df

# Streamlit App UI
st.set_page_config(page_title="Job Notifier", layout="wide")
st.title("💼 Job Notifier App (Karkidi.com)")
st.markdown("Automatically cluster and filter new job postings based on your skill preferences.")

# Sidebar for user input
with st.sidebar:
    st.header("🎯 Your Preferences")
    keyword = st.text_input("Enter job keyword", value="data science")
    pages = st.slider("Pages to scrape", min_value=1, max_value=3, value=1)
    user_clusters = st.multiselect("Select preferred clusters", options=[0, 1, 2, 3, 4], default=[0, 1])
    run_button = st.button("🔍 Fetch & Classify Jobs")

# Main logic when button is clicked
if run_button:
    with st.spinner("🔄 Scraping job postings..."):
        jobs_df = scrape_karkidi_jobs(keyword=keyword, pages=pages)

    if jobs_df.empty:
        st.warning("⚠️ No job postings found for the given keyword.")
    else:
        with st.spinner("🔍 Classifying jobs into clusters..."):
            model, vectorizer = load_models()
            classified_df = classify_new_jobs(jobs_df, model, vectorizer)

        matched_df = classified_df[classified_df["Cluster"].isin(user_clusters)]

        if not matched_df.empty:
            st.success(f"🎯 Found {len(matched_df)} jobs in selected clusters!")
            st.dataframe(matched_df[["Title", "Company", "Location", "Skills", "Cluster"]])

            csv = matched_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Matching Jobs", data=csv, file_name="matched_jobs.csv", mime="text/csv")
        else:
            st.info("✅ No jobs matched your selected clusters.")
