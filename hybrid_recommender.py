from data_loader import (
    movies_full
)

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

    # Skip the movie itself
    similarity_scores = similarity_scores[1:51]

    hybrid_scores = []

    for movie_idx, similarity_score in similarity_scores:

        movie_id = movies_full.iloc[
            movie_idx
        ]["movie_id"]

        movie_title = movies_full.iloc[
            movie_idx
        ]["title"]

        svd_score = model.predict(
            user_id,
            movie_id
        ).est

        # Normalize SVD score from [1,5] -> [0,1]
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