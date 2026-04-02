from __future__ import annotations

import ollama
import pandas as pd

from config import BATCH_SIZE, EMBED_MODEL


def embed_texts(texts, model = EMBED_MODEL):
    response = ollama.embed(model=model, input=texts)

    if "embeddings" not in response: # Error-ausgabe, wenn es ein problem mit den Embeddings gibt
        raise ValueError(f"Unerwartete Ollama-Antwort: {response}")

    embeddings = response["embeddings"]

    if len(embeddings) != len(texts): # Error-Ausgabe wenn mehr /weniger Embeddings kriert wurden, als es Texte gibt
        raise ValueError("Anzahl Embeddings passt nicht zur Anzahl Texte.")

    return embeddings


def add_embeddings_to_df(df, model = EMBED_MODEL, batch_size = BATCH_SIZE):
    df = df.copy()
    texts = df["Text"].tolist()

    all_embeddings = []

    for start in range(0, len(texts), batch_size): # so werden kleine Pakete (sog. batches) auf einmal vektorisiert (läuft schneller!)
        batch = texts[start:start + batch_size]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

    df["Embeddings"] = all_embeddings
    return df