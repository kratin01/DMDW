# 🚀 GetYourJob – Smart Job Finder

**GetYourJob** is an intelligent Streamlit-based job search application that helps users find relevant job opportunities from a dataset using keyword search, category prediction, and similarity-based ranking.

---

## 🔍 Features

- **📤 Upload Job Dataset**  
  Upload a CSV file containing job listings with columns: `id`, `title`, `description`, and `category`.

- **🔎 Inverted Index-Based Keyword Search**  
  Search for jobs using keywords. The app matches and ranks results using an inverted index built from job descriptions.

- **🧠 Category Prediction Using Machine Learning**  
  A `LinearSVC` classifier trained on TF-IDF vectors predicts the category of the job query.

- **📐 Cosine Similarity Matching**  
  Retrieves jobs most similar to the user query using Count Vectorization and cosine similarity.

- **🎛️ Sidebar Filters**  
  Filter results by predicted or user-selected job categories.

---

## 📁 Dataset Format

Your CSV file should contain the following columns:

| Column Name  | Description                    |
|--------------|--------------------------------|
| `id`         | Unique job identifier          |
| `title`      | Job title                      |
| `description`| Job description text           |
| `category`   | Job category label             |

---

## 🚀 How to Run

1. Install the required dependencies:

```bash
pip install streamlit pandas scikit-learn nltk
```

2. Launch the Streamlit app:

```bash
streamlit run app.py
```

3. Upload your dataset or use the default one included.

---

## 🧠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Data Handling**: Pandas
- **Text Processing**: NLTK's PorterStemmer, Regex
- **Machine Learning**: Scikit-learn (`TfidfVectorizer`, `LinearSVC`)
- **Similarity Search**: `CountVectorizer`, `cosine_similarity`

---

## 📌 Example Use Case

> A user searches for "Python developer"  
> 🔹 App predicts the category as `Software Development`  
> 🔹 Matches top jobs with keyword frequency  
> 🔹 Displays most relevant job postings using cosine similarity

---

## 💡 Future Improvements

- Add resume matching
- Include location and salary filters
- Enhance UI with interactive charts
- Add job bookmarking and saving

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue or suggest a feature.

---

## 📜 License

This project is licensed under the MIT License.

---
