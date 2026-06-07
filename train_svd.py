from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

from data_loader import (
    ratings,
    movies
)

# -------------------------
# Train SVD Model
# -------------------------

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        [
            "user_id",
            "movie_id",
            "rating"
        ]
    ],
    reader
)

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

model = SVD()

model.fit(trainset)

predictions = model.test(testset)

print(
    "RMSE:"
)

rmse(predictions)

# -------------------------
# Faster Lookup
# -------------------------

movie_lookup = dict(
    zip(
        movies["movie_id"],
        movies["title"]
    )
)

# -------------------------
# Collaborative Filtering
# -------------------------

def recommend_for_user(
    user_id,
    n=10
):

    rated_movies = set(
        ratings[
            ratings["user_id"]
            == user_id
        ]["movie_id"]
    )

    recommendations = []

    for movie_id in movies[
        "movie_id"
    ]:

        if movie_id not in rated_movies:

            predicted_rating = (
                model.predict(
                    user_id,
                    movie_id
                ).est
            )

            recommendations.append(
                (
                    movie_lookup[movie_id],
                    round(predicted_rating, 2)
                )
            )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:n]