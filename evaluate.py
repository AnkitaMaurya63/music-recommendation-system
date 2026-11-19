"""Evaluate the recommender with genre precision@K.

For a sample of songs we recommend K similar songs and check whether they share a
genre with the query song. To keep the test honest, genre is NOT used as an input
feature here (otherwise the model would simply be reading the answer).
A random baseline is reported for comparison.
"""
import argparse
import json

import numpy as np

from recommender import Recommender, load_songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    df = load_songs(args.data)
    print(f"Loaded {len(df)} unique songs")
    rec = Recommender(df, use_genre=False)

    rng = np.random.default_rng(42)
    queries = rng.choice(len(df), size=min(args.samples, len(df)), replace=False)

    model_p, random_p = [], []
    for q in queries:
        qg = df.at[q, "genres"]
        recs = rec.recommend(int(q), args.k)
        model_p.append(np.mean([bool(qg & df.at[r["id"], "genres"]) for r in recs]))
        rand = rng.choice(len(df), size=args.k, replace=False)
        random_p.append(np.mean([bool(qg & df.at[int(r), "genres"]) for r in rand]))

    result = {"songs": len(df), "queries": len(queries), "k": args.k,
              f"model_precision_at_{args.k}": round(float(np.mean(model_p)), 4),
              f"random_precision_at_{args.k}": round(float(np.mean(random_p)), 4)}
    print(json.dumps(result, indent=2))
    with open("results/metrics.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
