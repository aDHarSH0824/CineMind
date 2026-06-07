import pandas as pd

ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t",
    header=None
)

ratings.columns = [
    "user_id",
    "movie_id",
    "rating",
    "timestamp"
]

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    header=None,
    encoding="latin-1"
)

movies = movies[[0,1]]

movies.columns = [
    "movie_id",
    "title"
]

df = pd.merge(ratings, movies, on="movie_id")

movies_full = pd.read_csv(
    "ml-100k/u.item",
    sep="|",
    header=None,
    encoding="latin-1"
)

genres = pd.read_csv(
    "ml-100k/u.genre",
    sep="|",
    header=None
)

genre_names = genres[0].tolist()

column_names = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url"
] + genre_names

movies_full.columns = column_names

__all__ = [
    "ratings",
    "movies",
    "movies_full",
    "df",
    "genre_names",
    "movie_lookup",
    "title_lookup"
]

movie_lookup = dict(
    zip(
        movies["movie_id"],
        movies["title"]
    )
)

title_lookup = dict(
    zip(
        movies["title"],
        movies["movie_id"]
    )
)