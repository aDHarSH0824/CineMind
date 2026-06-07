from recommender import (
    recommend_popular_movies,
    recommend_similar_movies
)

from train_svd import (
    recommend_for_user
)

print("\nPOPULAR MOVIES\n")

print(
    recommend_popular_movies()
)

print(
    "\nSIMILAR TO TOY STORY\n"
)

print(
    recommend_similar_movies(
        "Toy Story (1995)"
    )
)

print(
    "\nUSER 10 RECOMMENDATIONS\n"
)

print(
    recommend_for_user(10)
)

from hybrid_recommender import (
    recommend_hybrid
)

print(
    "\nHYBRID RECOMMENDATIONS\n"
)

print(
    recommend_hybrid(
        10,
        "Toy Story (1995)"
    )
)