

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_karkidi_jobs(keywords=["data science", "data analyst", "data scientist", "software engineer"], pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for keyword in keywords:
        print(f"Searching for: {keyword}")
        for page in range(1, pages + 1):
            url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
            print(f"Scraping page {page} for keyword '{keyword}'")
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
                    print(f"Error parsing job block: {e}")
                    continue

            time.sleep(1)

    return pd.DataFrame(jobs_list)

# Usage
if __name__ == "__main__":
    df_jobs = scrape_karkidi_jobs(pages=5)
    print(df_jobs.head())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def determine_optimal_clusters(skill_matrix, max_k=10):
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(skill_matrix)
        score = silhouette_score(skill_matrix, labels)
        silhouette_scores.append(score)



    # Return the best k based on highest silhouette score
    best_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return best_k

def assign_clusters_to_jobs(job_dataframe):
    # Convert skills text into TF-IDF features
    tfidf = TfidfVectorizer(stop_words='english')
    skill_matrix = tfidf.fit_transform(job_dataframe['Skills'])

    # Determine the optimal number of clusters
    optimal_k = determine_optimal_clusters(skill_matrix, max_k=10)

    # Perform KMeans clustering using the optimal number of clusters
    clustering_model = KMeans(n_clusters=optimal_k, random_state=42)
    labels = clustering_model.fit_predict(skill_matrix)

    # Add cluster labels to the DataFrame
    job_dataframe['Cluster'] = labels

    return job_dataframe, clustering_model, tfidf

# Example usage:
clustered_jobs, model_used, tfidf_used = assign_clusters_to_jobs(df_jobs)
print(clustered_jobs.head())

import joblib

def persist_models(model, tfidf_vectorizer, model_path='kmeans_model.pkl', vectorizer_path='vectorizer.pkl'):

    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

# Example usage (use after fitting your model):
# Use the variables assigned in the previous cell
persist_models(model_used, tfidf_used)

def predict_job_cluster(skills_text, model, tfidf_vec):
    # Convert the input skills text into vector form using the TF-IDF vectorizer
    features = tfidf_vec.transform([skills_text])
    predicted_label = model.predict(features)
    return predicted_label[0]

# Load the saved model and vectorizer
model = joblib.load('kmeans_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Predict cluster for sample input
sample_skills = "AWS, Python, Data Science, Machine Learning"
cluster = predict_job_cluster(sample_skills, model, vectorizer)
print(f"Predicted cluster: {cluster}")

!pip install schedule

import schedule
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
import joblib

# Load persisted model and vectorizer
model = joblib.load('kmeans_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def email_alert(subject_line, message, recipient):
    sender = "your_email@example.com"
    sender_password = "your_password"
    smtp_server = smtplib.SMTP('smtp.example.com', 587)
    smtp_server.starttls()
    smtp_server.login(sender, sender_password)

    email_message = MIMEMultipart()
    email_message['From'] = sender
    email_message['To'] = recipient
    email_message['Subject'] = subject_line
    email_message.attach(MIMEText(message, 'plain'))

    smtp_server.sendmail(sender, recipient, email_message.as_string())
    smtp_server.quit()

def scrape_and_notify_daily():
    jobs_df = scrape_karkidi_jobs(keyword="data science", pages=2)

    # Use existing model and vectorizer to assign clusters
    skill_matrix = vectorizer.transform(jobs_df['Skills'])
    labels = model.predict(skill_matrix)
    jobs_df['Cluster'] = labels

    # Find jobs matching user interests
    interests = ["data science", "python","software engineer", "data analyst"]
    matched_jobs = jobs_df[jobs_df['Skills'].str.contains('|'.join(interests), case=False)]

    if not matched_jobs.empty:
        subj = "Job Updates Based on Your Interests"
        body_text = matched_jobs.to_string(index=False)
        email_alert(subj, body_text, "user@example.com")

# Schedule the daily task at 10 AM
schedule.every().day.at("10:00").do(scrape_and_notify_daily)

while True:
    schedule.run_pending()
    time.sleep(1)
