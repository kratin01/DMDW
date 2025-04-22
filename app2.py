import streamlit as st
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

# 🌟 Title & Theme
st.set_page_config(page_title="GetYourJob", layout="centered")
st.markdown("<h1 style='color:#4a7ebB;'>🚀 GetYourJob – Smart Job Finder</h1>", unsafe_allow_html=True)

# 📤 Upload Option
uploaded_file = st.file_uploader("Upload your job dataset (CSV with id, title, description, category)", type="csv")

@st.cache_data
def load_data(file):
    if file:
        return pd.read_csv(file)
    else:
        return pd.read_csv("sample_job_dataset.csv")

df = load_data(uploaded_file)

# ✅ Validate Columns
required_cols = {'id', 'title', 'description', 'category'}
if not required_cols.issubset(set(df.columns)):
    st.error("CSV must include columns: id, title, description, category")
    st.stop()

# 🧹 Preprocessing
stemmer = PorterStemmer()
def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return [stemmer.stem(word) for word in text.split()]

# 🔎 Inverted Index
inverted_index = defaultdict(set)
for _, row in df.iterrows():
    for word in preprocess(row["description"]):
        inverted_index[word].add(row["id"])

# 🧠 Train ML model using SVM
@st.cache_resource
def train_model(df):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["description"])
    y = df["category"]
    model = LinearSVC()
    model.fit(X, y)
    return tfidf, model

vectorizer, model = train_model(df)

# 🎯 Filters
with st.sidebar:
    st.header("🔧 Filters")
    selected_category = st.selectbox("Filter by category", ["All"] + sorted(df["category"].unique().tolist()))

# 🔍 Query Input
query = st.text_input("Enter job query (e.g., frontend, Python developer, marketing):")

if query:
    query_words = preprocess(query)
    matched_ids = defaultdict(int)
    for word in query_words:
        for job_id in inverted_index.get(word, []):
            matched_ids[job_id] += 1

    st.subheader("📌 Inverted Index Matches")
    if matched_ids:
        top_matches = sorted(matched_ids.items(), key=lambda x: x[1], reverse=True)[:5]
        for job_id, score in top_matches:
            job = df[df["id"] == job_id].iloc[0]
            st.markdown(f"**🔹 {job['title']} (Score: {score})**")
            st.caption(f"Category: {job['category']}")
            st.write(job['description'])
            st.markdown("---")
    else:
        st.info("No direct keyword matches found.")

    st.subheader("🧠 Predicted Job Category")
    query_vec = vectorizer.transform([query])
    predicted_cat = model.predict(query_vec)[0]
    st.success(f"Predicted Category: `{predicted_cat}`")

    # Filter by category if selected
    filtered_df = df[df["category"] == predicted_cat]
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]

    st.subheader("🤝 Top Cosine Similar Jobs")
    processed_desc = [" ".join(preprocess(d)) for d in filtered_df["description"]]
    processed_query = " ".join(preprocess(query))
    corpus = processed_desc + [processed_query]

    count_vec = CountVectorizer()
    vectors = count_vec.fit_transform(corpus)
    
    if vectors.shape[0] > 1:
        similarities = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        top_sim_indices = similarities.argsort()[::-1][:3]

        for idx in top_sim_indices:
            job = filtered_df.iloc[idx]
            sim_score = similarities[idx]
            st.markdown(f"**🔹 {job['title']} (Similarity: {round(sim_score, 2)})**")
            st.caption(f"Category: {job['category']}")
            st.write(job["description"])
            st.markdown("---")
    else:
        st.info("No similar jobs found based on the query.")
