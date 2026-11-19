"""FastAPI app: song search + recommendations.

Run:  uvicorn app:app --reload   ->  http://127.0.0.1:8000   (docs at /docs)
"""
import os

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from recommender import Recommender, load_songs

DATA_PATH = "data/dataset.csv"
app = FastAPI(title="Music Recommender API", version="1.0")

rec = None
if os.path.exists(DATA_PATH):
    rec = Recommender(load_songs(DATA_PATH), use_genre=True)


def _require():
    if rec is None:
        raise HTTPException(503, f"Dataset missing. Put the CSV at {DATA_PATH}")
    return rec


@app.get("/health")
def health():
    return {"status": "ok", "songs": 0 if rec is None else len(rec.df)}


@app.get("/search")
def search(q: str = Query(..., min_length=1, max_length=100), limit: int = 8):
    return _require().search(q, min(limit, 20))


@app.get("/random")
def random_song():
    """A random popular song id, used by the 'Surprise me' button."""
    r = _require()
    pool = r.df[r.df["popularity"] >= 50]
    pool = pool if len(pool) else r.df
    return {"id": int(pool.sample(1).index[0])}


@app.get("/recommend/{song_id}")
def recommend(song_id: int, n: int = 10):
    r = _require()
    if not 0 <= song_id < len(r.df):
        raise HTTPException(404, "Unknown song id")
    return {"song": r._row(song_id), "recommendations": r.recommend(song_id, min(n, 30))}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
