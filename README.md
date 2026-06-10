# 🎬 CineMind: Hybrid AI Movie Recommender

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Model SVD](https://img.shields.io/badge/SVD-Surprise%20Lib-blueviolet)](http://surpriselib.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CineMind** is a state-of-the-art, premium Movie Recommendation System powered by a Hybrid AI engine. Using the **MovieLens 100K dataset**, CineMind blends collaborative filtering (SVD) and content-based filtering (cosine similarity on genres) to deliver high-quality, personalized movie recommendations in a sleek, Netflix-inspired user interface.

---

## ✨ Features

- 🏠 **Unified Dashboard**: View platform-wide statistics (1,600+ movies, 943 active users, 100k+ ratings) and trending films.
- 🎬 **Hybrid AI Engine**: Recommends movies by calculating a weighted balance between content similarity (what the movie is about) and SVD collaborative matching (how likely a user is to rate the movie highly).
- 🎭 **Personalized Genre Explorer**: Filter recommendations by genre (e.g., Adventure, Sci-Fi) to see personalized user-profile picks alongside top-rated classics in that genre.
- ⚡ **Optimized Performance**: Features asynchronous poster fetching from the **OMDb API** with Streamlit memory caching (`@st.cache_data`) for instantaneous loading.
- 🎨 **Premium Visuals**: Custom dark glassmorphic styling, responsive movie-card grids, fluid hover transitions, and Outfit typography.

---

## ⚙️ Hybrid Recommendation Algorithm

The hybrid recommender scores candidates using two independent models:

1. **Content-Based Filtering**:
   - Represents movies as a binary genre vector.
   - Calculates **Cosine Similarity** between the selected movie and all other movies in the corpus.
   - $Similarity(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$

2. **Collaborative Filtering (SVD)**:
   - Fits a Singular Value Decomposition (SVD) model on user-item interaction histories (ratings).
   - Generates a personalized prediction value for the active user on candidate movies.
   - Normalizes SVD ratings to a $[0, 1]$ scale.

3. **Hybrid Score**:
   - Combines both components using a weighted linear combination:
   - $Hybrid\_Score = 0.4 \times Content\_Similarity + 0.6 \times Normalized\_SVD\_Score$

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with Custom HTML5 & CSS3
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (Cosine Similarity), [Surprise](http://surpriselib.com/) (SVD matrix factorization)
- **APIs**: [OMDb API](http://www.omdbapi.com/) (poster fetching & metadata extraction)

---

## 📂 Repository Structure

- `app.py`: The main entrypoint, containing the Streamlit UI dashboard and page routers.
- `hybrid_recommender.py`: Engine calculating similarity scores, SVD predictions, and handling genre sorting.
- `train_svd.py`: Trains the collaborative filtering model using the `surprise` SVD algorithm.
- `recommender.py`: Calculates popularity matrices and base cosine similarity scores.
- `data_loader.py`: Cleans, structures, and merges MovieLens dataset tables.
- `ml-100k/`: The raw MovieLens 100K dataset files.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have **Python 3.8+** installed.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aDHarSH0824/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit web interface using the local development server:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to explore the dashboard.

---

## 📈 Evaluation

- **Algorithm**: Singular Value Decomposition (SVD)
- **Metric**: Root Mean Squared Error (RMSE) on a 20% test-split
- **Result**: **RMSE ~0.93** (state-of-the-art performance for basic factorization models on ML-100K)
