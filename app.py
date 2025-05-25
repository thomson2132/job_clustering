import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import joblib
from sklearn.preprocessing import normalize

# -------------------- Scraper Function --------------------
def scrape_karkidi_jobs(keywords=["data science"], pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")

            job_blocks = soup.find_all("div", class_="ads-details")
            for job in job_blocks:
                try:
                    title = job.find("h4").get_text(strip=True)
                    company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                    location = job.find("p").get_text(strip=True)
                    experience = job.find("p", class_="emp-exp").get_text(strip=True)
                    key_skills_tag = job.find("span", string="Key Skills")
                    skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                    summary_tag = job.find("span", string="Summary")
                    summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                    jobs_list.append({
                        "Keyword": keyword,
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience,
                        "Summary": summary,
                        "Skills": skills
                    })
                except Exception as e:
                    continue

            time.sleep(1)

    return pd.DataFrame(jobs_list)

# -------------------- Load persisted ML components --------------------
@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return kmeans_model, vectorizer

# -------------------- Classify jobs using model & vectorizer --------------------
def classify_new_jobs(df, model, vectorizer):
    df = df.copy()
    df["Skills"] = df["Skills"].fillna("").str.lower()
    X = vectorizer.transform(df["Skills"])
    X = normalize(X)
    df["Cluster"] = model.predict(X)
    return df

# -------------------- Streamlit UI Setup --------------------
st.set_page_config(page_title="Job Notifier", layout="wide")
st.title("💼 Job Notifier App (Karkidi.com)")
st.markdown("Automatically cluster and filter new job postings based on your skill preferences.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("🎯 Your Preferences")
    keyword = st.text_input("Enter job keyword", value="data science")
    pages = st.slider("Pages to scrape", min_value=1, max_value=3, value=1)
    user_clusters = st.multiselect("Select preferred clusters", options=[0, 1, 2, 3, 4], default=[0, 1])
    run_button = st.button("🔍 Fetch & Classify Jobs")

# -------------------- Main Logic --------------------
if run_button:
    with st.spinner("🔄 Scraping job postings..."):
        jobs_df = scrape_karkidi_jobs(keywords=[keyword], pages=pages)

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
