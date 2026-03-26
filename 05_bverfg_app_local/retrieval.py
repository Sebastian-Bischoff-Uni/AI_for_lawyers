from __future__ import annotations

import math
from typing import List

import pandas as pd

from config import EMBED_MODEL
from embeddings import embed_texts


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def search_similar_passages(
    query: str,
    df: pd.DataFrame,
    model: str = EMBED_MODEL,
    top_k: int = 5,
) -> pd.DataFrame:
    if "Embeddings" not in df.columns:
        raise ValueError("Die Spalte 'Embeddings' fehlt.")

    if df["Embeddings"].isna().any():
        raise ValueError("Es fehlen Embeddings im DataFrame.")

    query_embedding = embed_texts([query], model=model)[0]

    result_df = df.copy()
    result_df["Score"] = result_df["Embeddings"].apply(
        lambda emb: cosine_similarity(query_embedding, emb)
    )

    return (
        result_df.sort_values("Score", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )