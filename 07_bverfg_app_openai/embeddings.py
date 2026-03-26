import os
from openai import OpenAI
import pandas as pd

from config import BATCH_SIZE, EMBED_MODEL


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY ist nicht gesetzt.")
    return OpenAI(api_key=api_key)


def embed_texts(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    client = get_openai_client()
    response = client.embeddings.create(
        model=model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def add_embeddings_to_df(
    df: pd.DataFrame,
    model: str = EMBED_MODEL,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    df = df.copy()
    texts = df["Text"].tolist()

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = embed_texts(batch, model=model)
        all_embeddings.extend(batch_embeddings)

    df["Embeddings"] = all_embeddings
    return df