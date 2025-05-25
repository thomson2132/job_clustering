import streamlit as st
import pandas as pd
import time
import joblib
import requests
from bs4 import BeautifulSoup

@st.cache_resource
def load_models():
    kmeans_model = joblib.load('kmeans_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return kmeans_model, vectorizer

def scrape_karkidi_jobs(keywords, pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        for keyword in keywords:
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
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience,
                        "Summary": summary,
                        "Skills": skills
                    })
                except Exception:
                    continue
            time.sleep(1)
    return pd.DataFrame(jobs_list)

def cluster_jobs_with_loaded_model(df, kmeans_model, vectorizer):
    X = vectorizer.transform(df['Skills'])
    df['Cluster'] = kmeans_model.predict(X)
    return df

st.title("Karkidi Job Scraper with Pre-trained Clustering")

keywords_input = st.text_input("Enter job keywords (comma-separated)", value="data science, software engineer, data analyst")
pages = st.number_input("Number of pages to scrape per keyword", min_value=1, max_value=5, value=2)

if st.button("Scrape & Cluster Jobs"):
    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]
    with st.spinner("Loading models..."):
        kmeans_model, vectorizer = load_models()
    with st.spinner(f"Scraping {pages} pages for keywords: {', '.join(keywords)} ..."):
        jobs_df = scrape_karkidi_jobs(keywords, pages)
    if jobs_df.empty:
        st.warning("No jobs found for these keywords.")
    else:
        clustered_df = cluster_jobs_with_loaded_model(jobs_df, kmeans_model, vectorizer)
        st.success("Jobs clustered successfully!")
        st.dataframe(clustered_df)
