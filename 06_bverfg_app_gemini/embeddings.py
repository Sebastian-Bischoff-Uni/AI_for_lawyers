from __future__ import annotations

import pandas as pd

from config import BATCH_SIZE, EMBED_MODEL
from gemini_client import get_gemini_client
from google.genai import types


def embed_texts(texts, model= EMBED_MODEL):
    """
    Erzeugt Embeddings mit Gemini.
    """
    if not texts:
        return []

    client = get_gemini_client()
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")     # hier bei gemini gibt es verschiedene Modi (SEMANTIC_SIMILARITY, RETRIEVAL_QUERY, FACT_VERIFICATION etc.)

    )

    if not hasattr(response, "embeddings") or response.embeddings is None:
        raise ValueError(f"Unerwartete Embedding-Antwort: {response!r}")

    embeddings = []

    for item in response.embeddings:
        values = getattr(item, "values", None)
        if values is None:
            raise ValueError(f"Embedding ohne values gefunden: {item!r}")
        embeddings.append(list(values))

    if len(embeddings) != len(texts):
        raise ValueError(
            f"Anzahl Embeddings ({len(embeddings)}) passt nicht zu "
            f"Anzahl Texte ({len(texts)})."
        )

    return embeddings


def add_embeddings_to_df(df, model= EMBED_MODEL, batch_size = BATCH_SIZE):
    """
    Fügt einem DataFrame in Batches Embeddings hinzu.
    """
    df = df.copy()
    texts = df["Text"].tolist()

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

    df["Embeddings"] = all_embeddings
    return df