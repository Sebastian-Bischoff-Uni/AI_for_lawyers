from __future__ import annotations

import ollama
import pandas as pd

from config import BATCH_SIZE, EMBED_MODEL


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)

    if "embeddings" not in response:
        raise ValueError(f"Unerwartete Ollama-Antwort: {response}")

    embeddings = response["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError("Anzahl Embeddings passt nicht zur Anzahl Texte.")

    return embeddings


def add_embeddings_to_df(
    df: pd.DataFrame,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    df = df.copy()
    texts = df["Text"].tolist()

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

    df["Embeddings"] = all_embeddings
    return df