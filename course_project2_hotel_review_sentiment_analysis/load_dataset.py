"""Load 2,000 sentences and their embeddings, plus 30 independent demo reviews.
Requires numpy. Run: python load_dataset.py
"""
from pathlib import Path
import csv
import numpy as np


def load_dataset(data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent
    def read_csv(name):
        with (root / name).open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    sentences = read_csv("hotel_sentences_2000.csv")
    demos = read_csv("hotel_demo_reviews_30.csv")
    with np.load(root / "hotel_embeddings_256.npz", allow_pickle=False) as data:
        X = data["embeddings"]
        sentence_ids = data["sentence_ids"].tolist()
        embedding_texts = data["sentences"].tolist()
    assert X.shape == (2000, 256) and X.dtype == np.float32
    assert np.isfinite(X).all()
    assert sentence_ids == [r["sentence_id"] for r in sentences]
    assert embedding_texts == [r["sentence"] for r in sentences]
    assert len(set(sentence_ids)) == 2000 and len(demos) == 30
    assert not ({r["review_id"] for r in demos} & {r["review_id"] for r in sentences})
    return X, sentences, demos


if __name__ == "__main__":
    X, sentences, demos = load_dataset()
    print("Sentence embeddings:", X.shape, X.dtype)
    print("First sentence:", sentences[0]["sentence_id"], sentences[0]["sentence"])
    print("Full demo reviews:", len(demos))
    print("First demo:", demos[0]["demo_id"], demos[0]["review_text"])
    print("Demo reviews are independent website examples, not labeled test data.")
    print("Split and embed demo text with the same encoder before prediction.")
