from __future__ import annotations

import streamlit as st
import pandas as pd

from config import BATCH_SIZE, EMBED_MODEL, TOP_K

from scraper import parse_bverfg_decision
from embeddings import add_embeddings_to_df
from rag import ask_rag


@st.cache_data() # Decorator von Streamlit - bewirkt, dass das Ergebnis der Funktion vorübergehend gespeichert wird
def build_vectordb_from_url(url: str) -> pd.DataFrame:
    df = parse_bverfg_decision(url)
    df = add_embeddings_to_df(df, model=EMBED_MODEL, batch_size=BATCH_SIZE)
    return df


def main() -> None:
    st.set_page_config(page_title="BVerfG-RAG", layout="wide")

    st.title("BVerfG-RAG mit Ollama")
    st.write(
        "URL eines BVerfG-Urteils eingeben, Frage stellen und Antwort samt Fundstellen abrufen."
    )

    url = st.text_input(
        "URL des BVerfG-Urteils",
        value="zum Beispiel: https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2025/03/rs20250326_2bvr150520.html",
    )

    question = st.text_area(
        "Frage an das Urteil",
        placeholder="Zum Beispiel: Unter welchen Voraussetzungen darf eine Ergänzungsabgabe nach Art. 106 Abs. 1 Nr. 6 GG erhoben werden?",
        height=120,
    )

    if st.button("Frage beantworten", type="primary"):
        if not url.strip():
            st.error("Bitte eine URL eingeben.")
            return

        if not question.strip():
            st.error("Bitte eine Frage eingeben.")
            return

        try:
            with st.spinner("Urteil wird geladen und indexiert..."):
                df = build_vectordb_from_url(url)

            with st.spinner("Frage wird beantwortet..."):
                result = ask_rag(question, df, top_k=TOP_K)

            st.subheader("Antwort")
            st.write(result["answer"])

            st.subheader("Abgerufene Fundstellen")
            hits = result["hits"]

            for i, row in hits.iterrows():
                with st.expander(
                    f"Treffer {i + 1} – Randnummer {row['Randnummer']} – Score {row['Score']:.4f}",
                    expanded=(i == 0),
                ):
                    st.markdown(f"**Randnummer:** {row['Randnummer']}")
                    st.markdown(f"**Score:** {row['Score']:.4f}")
                    st.write(row["Text"])

            with st.expander("Prompt anzeigen"):
                st.text(result["prompt"])

            with st.expander("DataFrame-Vorschau"):
                st.dataframe(
                    df[["Randnummer", "Text"]].head(20),
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Fehler: {e}")


if __name__ == "__main__":
    main()
    #Befehl in terminal: uv run streamlit run app.py
    # Beispiel-URL: https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2023/11/ls20231128_2bvl000813.html?nn=68020
    # passende Frage: Kann § 6 Abs. 5 EStG nach Rechtsprechung des BVerfG noch in verfassungskonformer Weise ausgelegt werden?