from __future__ import annotations

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import REQUEST_TIMEOUT


def fetch_html(url: str, timeout: int = REQUEST_TIMEOUT) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_bverfg_decision(url: str) -> pd.DataFrame:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")

    reasons_div = soup.find("div", class_="c-decision__reasons")
    if reasons_div is None:
        raise ValueError("Kein <div class='c-decision__reasons'> gefunden.")

    rows: list[dict] = []
    current_randnummer: int | None = None

    for tag in reasons_div.children:
        if not getattr(tag, "name", None):
            continue

        if tag.name == "p" and "is-anchor" in (tag.get("class") or []):
            rn_text = tag.get_text(strip=True)
            current_randnummer = int(rn_text) if rn_text.isdigit() else None
            continue

        if (
            tag.name == "p"
            and "justify" in (tag.get("class") or [])
            and current_randnummer is not None
        ):
            text = tag.get_text(" ", strip=True)
            rows.append(
                {
                    "Text": text,
                    "Embeddings": None,
                    "Randnummer": current_randnummer,
                }
            )

    if not rows:
        raise ValueError(
            "Keine Randnummer/Text-Paare in <div class='c-decision__reasons'> gefunden."
        )

    return pd.DataFrame(rows, columns=["Text", "Embeddings", "Randnummer"])