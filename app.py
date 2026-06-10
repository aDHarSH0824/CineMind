import streamlit as st
import requests
import re
import pandas as pd
from hybrid_recommender import (
    recommend_hybrid,
    recommend_by_genre_for_user,
    get_popular_movies_by_genre
)
from recommender import recommend_popular_movies
from train_svd import recommend_for_user
from data_loader import movies_full, ratings, genre_names

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CineMind - Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# POSTER FETCHING WITH CACHE
# -----------------------------
OMDB_API_KEY = "da59022a"

@st.cache_data(show_spinner=False)
def get_movie_poster(movie_title):
    try:
        # Clean title (remove year like (1995))
        clean_title = re.sub(r"\(\d{4}\)", "", movie_title).strip()
        url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={clean_title}"
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get("Poster") and data["Poster"] != "N/A":
                return data["Poster"]
    except Exception:
        pass
    return None

# -----------------------------
# CUSTOM CSS STYLE
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Global body override */
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0c0813 !important;
    background-image: radial-gradient(circle at 10% 20%, rgba(90, 18, 142, 0.12) 0%, rgba(10, 5, 20, 1) 90%) !important;
    color: #e2e8f0 !important;
}

/* Sidebar override */
[data-testid="stSidebar"] {
    background-color: #060309 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

.main-title {
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #e50914 0%, #ff3b30 50%, #ff9500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
    text-align: center;
}

.sub-title {
    font-size: 1.1rem !important;
    color: #94a3b8 !important;
    text-align: center;
    margin-bottom: 2rem !important;
}

/* Glassmorphism panel styling */
.glass-panel {
    background: rgba(255, 255, 255, 0.025);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}

.section-header {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    border-left: 4px solid #e50914;
    padding-left: 0.75rem;
    margin-bottom: 1.5rem !important;
    color: #ffffff !important;
}

/* Card layout grid */
.movie-card {
    background: rgba(18, 10, 28, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    height: 380px;
}

.movie-card:hover {
    transform: translateY(-6px);
    border-color: #e50914;
    box-shadow: 0 10px 20px rgba(229, 9, 20, 0.22);
    background: rgba(26, 15, 40, 0.85);
}

.card-poster-wrapper {
    position: relative;
    width: 100%;
    height: 250px;
    overflow: hidden;
    background: #000;
}

.movie-poster {
    width: 100%;
    height: 100%;
    object-fit: cover;
    transition: transform 0.4s ease;
}

.movie-card:hover .movie-poster {
    transform: scale(1.04);
}

.card-info {
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    flex-grow: 1;
}

.movie-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #ffffff;
    line-height: 1.3;
    text-align: center;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    height: 38px;
    margin-bottom: 6px;
}

.card-details {
    display: flex;
    justify-content: space-around;
    align-items: center;
    font-size: 0.72rem;
    background: rgba(0, 0, 0, 0.45);
    padding: 5px 8px;
    border-radius: 6px;
    color: #cbd5e1;
    font-weight: 500;
}

/* Custom styled inputs and buttons */
.stButton>button {
    background: linear-gradient(90deg, #e50914 0%, #ff3b30 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.2) !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4) !important;
    background: linear-gradient(90deg, #ff1a1a 0%, #ff5252 100%) !important;
}

/* Metric styling override */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.015) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    text-align: center !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #e50914 !important;
    background: linear-gradient(45deg, #e50914, #ff5b5b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #94a3b8 !important;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# HELPER: CARD GRID RENDERER
# -----------------------------
def render_movie_grid(movies_list):
    """
    Renders movie recommendations in a clean row-by-row 5-column grid.
    movies_list: list of tuples (title, score1, score2, etc.)
    """
    if not movies_list:
        st.info("No recommendations found.")
        return

    n_cols = 5
    for i in range(0, len(movies_list), n_cols):
        row_items = movies_list[i:i + n_cols]
        cols = st.columns(n_cols)
        for j, item in enumerate(row_items):
            title = item[0]
            score_val = item[1]
            
            # Formulate detail box depending on available data
            if len(item) == 4:  # Hybrid: (title, hybrid, similarity, svd)
                sim_score = item[2]
                svd_score = item[3]
                detail_html = (
                    f'<div class="card-details">'
                    f'<span>🎯 Match: {score_val*100:.0f}%</span>'
                    f'<span>⭐ SVD: {svd_score:.1f}</span>'
                    f'</div>'
                )
            elif len(item) == 3:  # Popular genre: (title, avg_rating, count)
                count = item[2]
                detail_html = (
                    f'<div class="card-details">'
                    f'<span>⭐ Rating: {score_val:.2f}</span>'
                    f'<span>👥 Votes: {count}</span>'
                    f'</div>'
                )
            else:  # Standard SVD: (title, score_val)
                detail_html = (
                    f'<div class="card-details">'
                    f'<span>⭐ Predicted: {score_val:.1f}</span>'
                    f'</div>'
                )
            
            poster_url = get_movie_poster(title)
            # Professional fallbacks from Unsplash representing cinema
            if not poster_url:
                poster_url = "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?q=80&w=300&auto=format&fit=crop"
                
            html_content = (
                f'<div class="movie-card">'
                f'<div class="card-poster-wrapper">'
                f'<img class="movie-poster" src="{poster_url}" />'
                f'</div>'
                f'<div class="card-info">'
                f'<div class="movie-title" title="{title}">{title}</div>'
                f'{detail_html}'
                f'</div>'
                f'</div>'
            )
            
            with cols[j]:
                st.markdown(html_content, unsafe_allow_html=True)



# -----------------------------
# SIDEBAR / CONTROLS
# -----------------------------
st.sidebar.markdown(f"""
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="color: #e50914; font-size: 2rem; margin: 0; font-weight: 800;">CineMind</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 5px;">Hybrid AI Movie Recommender</p>
    </div>
    <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0 20px 0;">
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Go To",
    ["🏠 Home / Dashboard", "🎬 Hybrid Recommender", "🎭 Genre Explorer"],
    label_visibility="collapsed"
)

st.sidebar.markdown('<div class="section-header" style="margin-top: 2rem;">User Settings</div>', unsafe_allow_html=True)

# Dynamic max users calculation
max_users = int(ratings["user_id"].max())
active_user_id = st.sidebar.number_input(
    "Active User ID",
    min_value=1,
    max_value=max_users,
    value=10,
    help=f"Enter a User ID between 1 and {max_users} to personalize suggestions."
)

num_recommendations = st.sidebar.slider(
    "Count",
    min_value=5,
    max_value=25,
    value=10,
    step=5
)

# -----------------------------
# PAGES ROUTING
# -----------------------------

if page == "🏠 Home / Dashboard":
    # Hero Title
    st.markdown('<h1 class="main-title">🎬 CineMind Movie Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Explore analytics of MovieLens 100k and receive curated content</p>', unsafe_allow_html=True)
    
    # Dataset statistics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Movies", f"{len(movies_full)}")
    with c2:
        st.metric("Active Users", f"{ratings['user_id'].nunique()}")
    with c3:
        st.metric("Total Ratings", f"{len(ratings):,}")
        
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    
    # 2 Tabs for Trending vs Personalized
    tab1, tab2 = st.tabs(["🔥 Top Trending Movies", "🎯 Personalized For You"])
    
    with tab1:
        st.markdown('<div class="section-header">Trending Across Platform</div>', unsafe_allow_html=True)
        popular_recs = recommend_popular_movies(num_recommendations)
        # Format popular_recs into: (title, avg_rating, count)
        formatted_popular = []
        for idx, row in popular_recs.iterrows():
            formatted_popular.append((idx, row["avg_rating"], int(row["rating_count"])))
            
        render_movie_grid(formatted_popular)
        
    with tab2:
        st.markdown(f'<div class="section-header">Personalized Picks for User #{active_user_id}</div>', unsafe_allow_html=True)
        with st.spinner("Analyzing user ratings profile..."):
            personalized_recs = recommend_for_user(active_user_id, num_recommendations)
            render_movie_grid(personalized_recs)

elif page == "🎬 Hybrid Recommender":
    st.markdown('<h1 class="main-title">🎬 Hybrid AI Recommendations</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Combining collaborative filters (SVD) with content similarity (Genres)</p>', unsafe_allow_html=True)
    
    # Form layout
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-bottom: 0.5rem !important;">Select Reference Movie</div>', unsafe_allow_html=True)
        
        movie_title = st.selectbox(
            "Reference Movie",
            sorted(movies_full["title"].unique()),
            label_visibility="collapsed"
        )
        
        st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
        btn_clicked = st.button("🚀 Find Similar Hybrid Matches", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    if btn_clicked:
        st.markdown(f'<div class="section-header">Because you watched "{movie_title}"</div>', unsafe_allow_html=True)
        with st.spinner("Running hybrid matching logic..."):
            try:
                # Returns: (movie_title, final_score, similarity_score, svd_score)
                hybrid_recs = recommend_hybrid(active_user_id, movie_title, num_recommendations)
                render_movie_grid(hybrid_recs)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

elif page == "🎭 Genre Explorer":
    st.markdown('<h1 class="main-title">🎭 Explore by Genre</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Find trending and personalized movies in your favorite genres</p>', unsafe_allow_html=True)
    
    # Clean available genre list
    display_genres = sorted([g for g in genre_names if g != 'unknown'])
    
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown('<div class="section-header" style="margin-bottom: 0.5rem !important;">Choose Your Theme</div>', unsafe_allow_html=True)
        selected_genre = st.selectbox(
            "Select Genre",
            display_genres,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    tab1, tab2 = st.tabs([f"🎯 Personalized {selected_genre} Picks", f"🔥 Popular {selected_genre} Movies"])
    
    with tab1:
        st.markdown(f'<div class="section-header">Personalized {selected_genre} for User #{active_user_id}</div>', unsafe_allow_html=True)
        with st.spinner(f"Filtering {selected_genre} with user profile SVD..."):
            try:
                # Returns (title, predicted_rating)
                genre_recs = recommend_by_genre_for_user(active_user_id, selected_genre, num_recommendations)
                render_movie_grid(genre_recs)
            except Exception as e:
                st.error(f"Error fetching recommendations: {str(e)}")
                
    with tab2:
        st.markdown(f'<div class="section-header">Top Trending {selected_genre} Movies</div>', unsafe_allow_html=True)
        with st.spinner(f"Aggregating trending {selected_genre} ratings..."):
            try:
                # Returns (title, avg_rating, rating_count)
                genre_popular = get_popular_movies_by_genre(selected_genre, num_recommendations)
                render_movie_grid(genre_popular)
            except Exception as e:
                st.error(f"Error fetching trending movies: {str(e)}")