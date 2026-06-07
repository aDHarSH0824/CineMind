import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import (
    df,
    movies_full,
    genre_names
)

# -------------------------
# Popularity Recommender
# -------------------------

avg_rating = df.groupby("title")["rating"].mean()

rating_count = df.groupby("title")["rating"].count()

movie_stats = pd.DataFrame()

movie_stats["avg_rating"] = avg_rating
movie_stats["rating_count"] = rating_count


def recommend_popular_movies(n=10):

    popular_movies = movie_stats[
        movie_stats["rating_count"] >= 100
    ]

    popular_movies = popular_movies.sort_values(
        by="avg_rating",
        ascending=False
    )

    return popular_movies.head(n)


# -------------------------
# Content Based Recommender
# -------------------------

feature_matrix = movies_full[genre_names]

similarity_matrix = cosine_similarity(
    feature_matrix
)

movie_indices = pd.Series(
    movies_full.index,
    index=movies_full["title"]
).drop_duplicates()


def recommend_similar_movies(
    movie_name,
    n=5
):

    idx = movie_indices[movie_name]

    similarity_scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:n+1]

    movie_ids = [
        movie[0]
        for movie in similarity_scores
    ]

    return movies_full[
        "title"
    ].iloc[movie_ids]