import streamlit as st
import pandas as pd
import joblib

# Load the saved model and TF-IDF vectorizer
model = joblib.load('kmeans_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Predict cluster for a single skills input
def predict_job_cluster(skills_text, model, tfidf_vec):
    features = tfidf_vec.transform([skills_text])
    predicted_label = model.predict(features)
    return predicted_label[0]

# Assign clusters to job DataFrame using loaded model
def assign_clusters_to_jobs(job_dataframe, model, tfidf_vec):
    skill_matrix = tfidf_vec.transform(job_dataframe['Skills'])
    labels = model.predict(skill_matrix)
    job_dataframe['Cluster'] = labels
    return job_dataframe

# Streamlit app layout
st.title("Job Clustering App with Streamlit")
st.write("Upload a CSV with a `Skills` column to cluster jobs using a pre-trained model.")

# File uploader
uploaded_file = st.file_uploader("Upload your jobs CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        df_jobs = pd.read_csv(uploaded_file)
        if 'Skills' not in df_jobs.columns:
            st.error("Uploaded CSV must contain a 'Skills' column.")
        else:
            clustered_df = assign_clusters_to_jobs(df_jobs, model, vectorizer)
            st.success("Jobs clustered successfully!")
            st.write(clustered_df)

            csv_download = clustered_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered CSV", data=csv_download, file_name="clustered_jobs.csv", mime="text/csv")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")
st.subheader("Predict Cluster for Custom Skills Input")
input_skills = st.text_input("Enter skills (comma-separated):", placeholder="e.g. Python, AWS, Machine Learning")

if st.button("Predict Cluster"):
    if input_skills.strip() == "":
        st.warning("Please enter some skills.")
    else:
        cluster = predict_job_cluster(input_skills, model, vectorizer)
        st.success(f"The predicted cluster for the entered skills is: **Cluster {cluster}**")
