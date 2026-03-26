from __future__ import annotations

import ollama
import pandas as pd

from config import CHAT_MODEL, EMBED_MODEL, TOP_K
from retrieval import search_similar_passages


def build_context_from_hits(hits: pd.DataFrame) -> str:
    context_parts = []

    for i, row in hits.iterrows():
        randnummer = row["Randnummer"]
        text = row["Text"]
        score = row.get("Score", None)

        part = (
            f"[Quelle {i + 1} | Randnummer {randnummer}"
            + (f" | Score {score:.4f}" if score is not None else "")
            + "]\n"
            f"{text}"
        )
        context_parts.append(part)

    return "\n\n".join(context_parts)


def build_rag_prompt(query: str, hits: pd.DataFrame) -> str:
    context = build_context_from_hits(hits)

    prompt = f"""
Du bist ein ein Experte für das deutsche Verfassungsrecht und spezialisiert auf die Analyse von Entscheidungen des Bundesverfassungsgerichts.

Du erhältst nachfolgend eine Frage sowie relevante Fundstellen aus einer Entscheidung des Bundesverfassungsgerichts (BVerfG), die du für die Beantwortung der Frage verwenden musst.
Beantworte die nachfolgend gestellte Frage ausschließlich auf Grundlage der unten angegebenen Fundstellen.
Wenn die Fundstellen für eine sichere Antwort nicht ausreichen, sage das ausdrücklich.
Zitiere in deiner Antwort möglichst die Randnummern in Klammern, z. B. (Rn. 12).
Beachte, dass Fundstellen in denen rechtsauffassungen im Konjunktiv dargestellt werden, nicht zwingend die Auffassung des Gerichts widerspiegeln, sondern auch Auffassung einer der Parteien sein können.

Frage:
{query}

Relevante Fundstellen:
{context}

Antworte präzise, juristisch sauber und gut strukturiert.
"""
    return prompt.strip()


def ask_rag(
    query: str,
    df: pd.DataFrame,
    retrieval_model: str = EMBED_MODEL,
    chat_model: str = CHAT_MODEL,
    top_k: int = TOP_K,
) -> dict:
    hits = search_similar_passages(
        query=query,
        df=df,
        model=retrieval_model,
        top_k=top_k,
    )

    prompt = build_rag_prompt(query, hits)

    response = ollama.chat(
        model=chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du beantwortest Fragen zu BVerfG-Entscheidungen "
                    "ausschließlich auf Basis des bereitgestellten Kontexts."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response["message"]["content"]

    return {
        "query": query,
        "answer": answer,
        "hits": hits,
        "prompt": prompt,
    }