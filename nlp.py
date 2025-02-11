import streamlit as st
import pandas as pd
import numpy as np
import os
from pypdf import PdfReader
import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Function to extract text from PDFs
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    sentences = sent_tokenize(text)
    features = {'feature': ""}
    stop_words = set(stopwords.words("english"))

    for sent in sentences:
        words = word_tokenize(sent)
        words = [word for word in words if word not in stop_words]
        tagged_words = pos_tag(words)
        filtered_words = [word for word, tag in tagged_words if tag not in ['DT', 'IN', 'TO', 'PRP', 'WP']]
        features['feature'] += " ".join(filtered_words) + " "
    return features['feature']

# Streamlit UI
st.set_page_config(page_title="Resume Screening App", layout="wide")
st.title("📄 Resume Screening using NLP with Data Visualization")

# Add logo to the sidebar
st.sidebar.image('nlplogo.jpg', width=150)

st.sidebar.header("Upload Resumes & Job Descriptions")

# Upload resumes (multiple PDFs)
uploaded_resumes = st.sidebar.file_uploader("Upload Resume PDFs", accept_multiple_files=True, type=["pdf"])

# Upload job descriptions (CSV)
uploaded_job_desc = st.sidebar.file_uploader("Upload Job Descriptions CSV", type=["csv"])

if uploaded_resumes and uploaded_job_desc:
    # Read job descriptions
    job_desc_df = pd.read_csv(uploaded_job_desc)
    
    # Ensure 'Features' column exists
    if "Features" not in job_desc_df.columns:
        job_desc_df["Features"] = job_desc_df["job_description"].apply(preprocess_text)
    
    # Extract text from uploaded PDFs
    resume_data = []
    for uploaded_file in uploaded_resumes:
        text = extract_text_from_pdf(uploaded_file)
        processed_text = preprocess_text(text)
        resume_data.append({"ID": uploaded_file.name, "Feature": processed_text})
    
    resume_df = pd.DataFrame(resume_data)
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words="english", max_features=800)
    tfidf_train_vectors = tfidf.fit_transform(resume_df["Feature"])
    tfidf_jobDesc_vectors = tfidf.transform(job_desc_df["Features"])

    # Compute Cosine Similarity
    similarity_matrix = cosine_similarity(tfidf_jobDesc_vectors.toarray(), tfidf_train_vectors.toarray())
    
    # Get the highest similarity resume for each job position
    highest_similarity = []
    for i in range(similarity_matrix.shape[0]):
        best_resume_index = np.argmax(similarity_matrix[i])
        highest_similarity.append({
            "Job Position": job_desc_df["position_title"].iloc[i],
            "Best Resume ID": resume_df["ID"].iloc[best_resume_index],
            "Similarity Score": similarity_matrix[i, best_resume_index]
        })
    
    best_matches_df = pd.DataFrame(highest_similarity)
    
    # Display Results
    st.subheader("Top Matching Resume for Each Job Position")
    st.dataframe(best_matches_df)
    
    # Display Similarity Heatmap
    st.subheader("Similarity Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    plt.xlabel("Resumes")
    plt.ylabel("Job Descriptions")
    st.pyplot(fig)
    
    # Visualize Distribution of Similarity Scores
    st.subheader("Distribution of Resume Similarity Scores")
    plt.figure(figsize=(8,5))
    sns.histplot(best_matches_df["Similarity Score"], bins=10, kde=True, color="blue")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    plt.title("Distribution of Resume Similarity Scores")
    st.pyplot(plt)

st.sidebar.markdown("---")
st.sidebar.info("Use the sidebar to upload resumes (PDF) and job descriptions (CSV).")
