from data_loader import movies_full

from recommender import (
    similarity_matrix,
    movie_indices
)

from train_svd import model


def recommend_hybrid(
    user_id,
    movie_name,
    n=10
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

    similarity_scores = similarity_scores[1:51]

    hybrid_scores = []

    for movie_idx, similarity_score in similarity_scores:

        movie_id = movies_full.iloc[movie_idx]["movie_id"]

        movie_title = movies_full.iloc[movie_idx]["title"]

        svd_score = model.predict(
            user_id,
            movie_id
        ).est

        normalized_svd_score = (
            svd_score - 1
        ) / 4

        final_score = (
            0.4 * similarity_score
            +
            0.6 * normalized_svd_score
        )

        hybrid_scores.append(
            (
                movie_title,
                round(final_score, 4),
                round(similarity_score, 4),
                round(svd_score, 4)
            )
        )

    hybrid_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return hybrid_scores[:n]


def get_hybrid_recommendations(
    movie_name,
    user_id=10,
    n=10
):
    return recommend_hybrid(
        user_id=user_id,
        movie_name=movie_name,
        n=n
    )


def recommend_by_genre_for_user(
    user_id,
    genre_name,
    n=10
):
    import pandas as pd
    from train_svd import model, ratings
    
    genre_movies = movies_full[movies_full[genre_name] == 1]
    
    # Get movies the user has already rated
    rated_movie_ids = set(ratings[ratings["user_id"] == user_id]["movie_id"])
    
    recommendations = []
    for _, row in genre_movies.iterrows():
        movie_id = row["movie_id"]
        movie_title = row["title"]
        
        # Predict rating
        pred_rating = model.predict(user_id, movie_id).est
        
        recommendations.append(
            (
                movie_title,
                round(pred_rating, 2)
            )
        )
        
    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )
    
    return recommendations[:n]


def get_popular_movies_by_genre(
    genre_name,
    n=10
):
    import pandas as pd
    from recommender import movie_stats
    
    genre_movies = movies_full[movies_full[genre_name] == 1]
    
    merged = pd.merge(
        genre_movies,
        movie_stats,
        on="title"
    )
    
    # Filter for movies with at least 20 ratings for credibility
    popular = merged[merged["rating_count"] >= 20]
    
    popular = popular.sort_values(
        by="avg_rating",
        ascending=False
    )
    
    recommendations = []
    for _, row in popular.head(n).iterrows():
        recommendations.append(
            (
                row["title"],
                round(row["avg_rating"], 2),
                int(row["rating_count"])
            )
        )
        
    return recommendations


if __name__ == "__main__":

    movie_name = "Toy Story (1995)"

    recommendations = get_hybrid_recommendations(
        movie_name=movie_name,
        user_id=10
    )

    print("\nHYBRID RECOMMENDATIONS\n")

    for movie in recommendations:
        print(movie)