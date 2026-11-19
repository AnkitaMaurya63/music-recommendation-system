"""Content-based music recommender.

Each song is represented by
  * its audio features (danceability, energy, tempo, valence, ...), min-max scaled, and
  * TF-IDF of its text metadata (artist, track name and, optionally, genre).
Songs are ranked by cosine similarity to the chosen song.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, normalize

AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo"]


def load_songs(path: str = "data/dataset.csv") -> pd.DataFrame:
    """Load the Kaggle 'Spotify Tracks Dataset' and clean it."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["track_id", "track_name", "artists"] + AUDIO)
    # the same track can be listed under several genres: keep them all as a set
    genre_sets = df.groupby("track_id")["track_genre"].agg(lambda s: set(s))
    df = df.drop_duplicates("track_id").copy()
    df["genres"] = df["track_id"].map(genre_sets)
    df = df.drop_duplicates(["track_name", "artists"]).reset_index(drop=True)
    return df


class Recommender:
    def __init__(self, df: pd.DataFrame, use_genre: bool = True,
                 w_audio: float = 1.0, w_text: float = 0.6):
        self.df = df
        text = df["artists"].fillna("") + " " + df["track_name"].fillna("")
        if use_genre:
            text = text + " " + df["genres"].apply(lambda g: " ".join(sorted(g)))
        tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z0-9']+", max_features=30000)
        text_vec = normalize(tfidf.fit_transform(text))
        audio_vec = normalize(MinMaxScaler().fit_transform(df[AUDIO]))
        self.X = normalize(
            hstack([csr_matrix(audio_vec * w_audio), text_vec * w_text]).tocsr())

    def search(self, query: str, limit: int = 8):
        q = query.strip().lower()
        if not q:
            return []
        mask = (self.df["track_name"].str.lower().str.contains(q, regex=False)
                | self.df["artists"].str.lower().str.contains(q, regex=False))
        hits = self.df[mask].sort_values("popularity", ascending=False).head(limit)
        return [self._row(i) for i in hits.index]

    def recommend(self, idx: int, n: int = 10):
        sims = (self.X @ self.X[idx].T).toarray().ravel()
        sims[idx] = -1
        top = np.argpartition(-sims, n)[:n]
        top = top[np.argsort(-sims[top])]
        return [dict(self._row(i), similarity=round(float(sims[i]), 4)) for i in top]

    def _row(self, i: int):
        r = self.df.iloc[i]
        return {"id": int(i), "track": r["track_name"], "artist": r["artists"],
                "genre": ", ".join(sorted(r["genres"])[:2]),
                "tempo": round(float(r["tempo"])), "popularity": int(r["popularity"])}
