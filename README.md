# Music Recommendation System

A content-based music recommender. Search for a song or artist, pick a song, and get similar tracks ranked by cosine similarity. It comes with a FastAPI REST API and a web UI.

## Dataset
[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle, about 114k tracks, 125 genres). Download `dataset.csv` and place it in the `data/` folder. It has audio features (danceability, energy, valence, tempo, ...), artist, track name and genre.

## Approach
1. **Cleaning** (`recommender.py`): drop rows with missing values, remove duplicate tracks, keep all genres a track appears under.
2. **Audio features**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence and tempo, scaled with Min-Max and L2-normalised.
3. **Text features**: TF-IDF over artist name, track name and genre.
4. **Similarity**: the two feature blocks are weighted and concatenated; songs are ranked by cosine similarity to the chosen song.

## Evaluation
Content-based recommenders have no single "accuracy", so I measure **genre precision@10**: for 500 random songs, the share of the top 10 recommendations that share a genre with the query song. Genre is excluded from the features during this test so the model cannot read the answer. A random baseline is reported alongside.

Run `python evaluate.py` and fill in your numbers (also saved in `results/metrics.json`):

| Metric | Value |
|---|---|
| Unique songs | _fill_ |
| Model precision@10 | _fill_ |
| Random baseline precision@10 | _fill_ |

## How to run
```bash
git clone https://github.com/AnkitaMaurya63/music-recommendation-system.git
cd music-recommendation-system
pip install -r requirements.txt
# put dataset.csv in data/
python evaluate.py
uvicorn app:app --reload
```
Open http://127.0.0.1:8000 for the web UI and http://127.0.0.1:8000/docs for the API.

## API
- `GET /search?q=<text>` returns matching songs
- `GET /recommend/{song_id}?n=10` returns the most similar songs with similarity scores

## Project structure
```
music-recommendation-system/
├── recommender.py   # data cleaning, features, similarity
├── evaluate.py      # precision@K evaluation vs random baseline
├── app.py           # FastAPI app
├── static/index.html
├── data/            # place dataset.csv here
└── results/         # metrics.json
```

## Tech stack
Python, pandas, NumPy, scikit-learn (TF-IDF, cosine similarity), SciPy, FastAPI, HTML/CSS/JavaScript.

## Limitations and ideas
- Content-based, so it does not learn from user listening history.
- Lyrics are not used; adding lyric embeddings or a collaborative-filtering signal would improve it.
