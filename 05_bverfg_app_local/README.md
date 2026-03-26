# BVerfG-RAG mit Streamlit und Ollama

## Überblick

Dieses Projekt ist eine kleine RAG-Anwendung (*Retrieval-Augmented Generation*) für Entscheidungen des Bundesverfassungsgerichts.

Die Anwendung hat zwei Eingabefelder:

* eine **URL** zu einer BVerfG-Entscheidung,
* eine **Frage** zu dieser Entscheidung.

Anschließend passiert Folgendes:

1. Die HTML-Seite der Entscheidung wird geladen.
2. Der Urteilstext wird absatzweise aus dem HTML-Code extrahiert.
3. Die einzelnen Absätze werden zusammen mit ihrer **Randnummer** in ein `pandas.DataFrame` geschrieben.
4. Für jeden Absatz wird mit einem lokalen Embedding-Modell (`qwen`) über **Ollama** ein Vektor erzeugt.
5. Zur Nutzerfrage werden die **Top-k ähnlichsten Absätze** per `Cosine Similarity` gefunden.
6. Diese Fundstellen werden als Kontext an das Sprachmodell **`gemma3`** übergeben.
7. Das Modell erzeugt daraus eine Antwort.
8. In der Streamlit-App (Frontend) werden anschließend

   * zuerst die **Antwort**,
   * dann die **abgerufenen Fundstellen** angezeigt.

---

## Ziel des Projekts

Das Ziel ist, aus einzelnen BVerfG-Urteilen ein lokales, abfragbares RAG-System zu bauen.

Die Anwendung soll vor allem:

* Urteile automatisiert über eine URL einlesen,
* diese strukturiert absatzweise erfassen,
* die relevanten Randnummern für eine Frage finden,
* und auf dieser Grundlage eine juristisch verwertbare Antwort erzeugen.

---

## Verwendete Python-Libraries

### Streamlit

Streamlit bildet die Benutzeroberfläche (Frontend).

Es gibt dort zwei Nutzereingaben:

* URL des Urteils
* juristische Frage

Außerdem zeigt Streamlit die Antwort, die Fundstellen und optional den Prompt bzw. eine DataFrame-Vorschau an.

### BeautifulSoup

BeautifulSoup wird zum Parsen des HTML-Dokuments verwendet.

Im konkreten BVerfG-HTML liegt der relevante Text in:

```html
<div class="c-decision__reasons">
```

Dort werden die Randnummern über

```html
<p class="is-anchor" id="8">8</p>
```

erkannt, und die zugehörigen Textabsätze stehen in

```html
<p class="justify">...</p>
```

### pandas

Die geparsten Entscheidungsabsätze werden in einem `DataFrame` gespeichert.

Geplante bzw. verwendete Struktur:

* `Text`
* `Embeddings`
* `Randnummer`

### Ollama

Ollama übernimmt zwei Aufgaben:

1. **Embeddings** mit einem Embedding-Modell, etwa:

   * `qwen3-embedding:8b` (oder andere Modelle wie 0.6b)
2. **Antwortgenerierung** mit dem Sprachmodell:

   * `gemma3:12b`

### Cosine Similarity

Die Ähnlichkeit zwischen Nutzerfrage und Entscheidungsabsätzen wird über Cosine Similarity bestimmt.

Dadurch können die relevantesten Fundstellen aus dem Urteil gefunden werden.

---

## Projektstruktur

Das Projekt ist in mehrere Python-Dateien aufgeteilt, damit die Logik sauber getrennt bleibt.

```text
05_bverfg_app_local/
├── app.py
├── config.py
├── scraper.py
├── embeddings.py
├── retrieval.py
└── rag.py
```

---

## Beschreibung der einzelnen Dateien

## `config.py`

Diese Datei enthält zentrale Konfigurationswerte.

Typische Inhalte:

* Name des Embedding-Modells
* Name des Chat-Modells
* `TOP_K` (der top-k-Wert gibt die Anzahl k der ähnlichsten Urteilsabsätze aus, die letztlich dem Prompt angefügt werden)
* Batch-Größe für Embeddings (die Batch-Größe gibt an, wie viele Urteilsabsätze gleichzeitig an das Embedding-Modell übergeben und in einem Verarbeitungsschritt vektorisiert werden)
* Timeout für HTTP-Requests

Beispiel:

```python
EMBED_MODEL = "mxbai-embed-large"
CHAT_MODEL = "gemma3:12b"
TOP_K = 5
BATCH_SIZE = 16
REQUEST_TIMEOUT = 30
```

Vorteil: Modellnamen und Standardwerte sind an einer Stelle zentral änderbar.

---

## `scraper.py`

Diese Datei enthält Funktionen zum Laden und Parsen einer BVerfG-Entscheidung.

### Aufgaben

* HTML einer URL abrufen
* den Container `div.c-decision__reasons` finden
* Randnummern erkennen
* zugehörige Textabsätze extrahieren
* ein `DataFrame` erzeugen

### Wichtige Funktionen

#### `fetch_html(url)`

Lädt die HTML-Seite über die library `requests`.

#### `parse_bverfg_decision(url)`

Parst die Entscheidung und erzeugt ein `DataFrame` mit den Spalten:

* `Text`
* `Embeddings`
* `Randnummer`

### Vorgehen des Parsers

Die Funktion läuft die direkten "Kinder" von `div.c-decision__reasons` in Reihenfolge durch.

* Wenn ein Tag ein `p.is-anchor` ist, wird die Randnummer gespeichert.
* Wenn danach ein `p.justify` folgt, wird sein Text als Absatz unter dieser Randnummer gespeichert.

Damit wird die Struktur des Urteils absatzweise sauber abgebildet.

---

## `embeddings.py`

Diese Datei ist für die Vektorisierung der Texte zuständig.

### Aufgaben

* Texte werden von einem über Ollama abgerufenen Embeddingmodell in den Vektorraum "eingebettet"
* Embeddings werden in ein DataFrame geschrieben

### Wichtige Funktionen

#### `embed_texts(texts, model=EMBED_MODEL)`

Erzeugt Embeddings für eine Liste von Texten.

Die Funktion verwendet:

```python
ollama.embed(model=model, input=texts)
```

#### `add_embeddings_to_df(df, model=..., batch_size=...)`

Nimmt das DataFrame und füllt die Spalte `Embeddings` aus.

Das geschieht in Batches, damit größere Urteile effizient verarbeitet werden.

---

## `retrieval.py`

Diese Datei enthält die Funktionen zum datenabruf aus dem Dataframe (**Retrieval**)

### Aufgaben

* für eine Frage die ähnlichsten Passagen finden
* dafür die Cosine Similarity berechnen

### Wichtige Funktionen

#### `cosine_similarity(a, b)`

Berechnet die Ähnlichkeit zwischen zwei Embedding-Vektoren.

#### `search_similar_passages(query, df, model=..., top_k=5)`

Ablauf:

1. Die Frage wird vektorisiert.
2. Für jede Passage im DataFrame wird die Cosine Similarity zur gestellten Frage berechnet.
3. Die Passagen werden nach Score sortiert.
4. Die Top-`k`-Treffer werden zurückgegeben.

Das Ergebnis ist der oben erzeugte `DataFrame`, nun ergänzt um eine Spalte `Score`.

---

## `rag.py`

Diese Datei verbindet  Datenabruf/Retrieval und Sprachmodell.

### Aufgaben

* aus den gefundenen Passagen einen Kontext bauen
* daraus einen RAG-Prompt erzeugen
* das Sprachmodell aufrufen

### Wichtige Funktionen

#### `build_context_from_hits(hits)`

Erzeugt aus den Top-Treffern einen lesbaren Kontextblock.

Beispielstruktur:

```text
[Quelle 1 | Randnummer 100 | Score 0.8123]
Text der Fundstelle ...
```

#### `build_rag_prompt(query, hits)`

Baut den eigentlichen Prompt für das Sprachmodell.

Der Prompt enthält:

* die juristische Rolle des Modells,
* die Anweisung, nur auf Basis der Fundstellen zu antworten,
* die Nutzerfrage,
* die abgerufenen Fundstellen.

#### `ask_rag(query, df, retrieval_model=..., chat_model=..., top_k=...)`

Das ist die zentrale RAG-Funktion.

Ablauf:

1. Top-5-Fundstellen suchen
2. Prompt erzeugen
3. `gemma3:12b` über `ollama.chat(...)` aufrufen
4. Antwort und Treffer zurückgeben

Rückgabe als `dict` mit:

* `query`
* `answer`
* `hits`
* `prompt`

---

## `app.py`

Diese Datei führtalle anderen Dateien zusammen und ist der Einstiegspunkt der Streamlit-Anwendung.

Sie enthält keine eigenen "Backend-Funktionen" mehr, sondern nur noch die Benutzeroberfläche und den Aufruf der übrigen Module.

### Aufgaben

* Seitenlayout erzeugen
* URL und Frage entgegennehmen
* Urteil laden und indexieren
* Frage beantworten
* Antwort und Fundstellen anzeigen

### Typischer Ablauf in `app.py`

1. URL-Feld anzeigen
2. Fragefeld anzeigen
3. Beim Klick auf „Frage beantworten“:

   * Urteil scrapen
   * Embeddings erzeugen
   * RAG ausführen
4. Antwort rendern
5. Fundstellen anzeigen
6. optional Prompt und DataFrame-Vorschau anzeigen

### Caching

Cache ist ein Zwischenspeicher
Mit

```python
@st.cache_data
```

kann das eingelesene und eingebettete Urteil zwischengespeichert werden, damit es bei wiederholten Fragen nicht jedes Mal neu verarbeitet werden muss.

---

## Datenfluss im gesamten System

Der Ablauf des Programms lässt sich so zusammenfassen:

```text
URL -> HTML laden -> Entscheidungsgründe parsen -> DataFrame erzeugen
   -> Embeddings für Absätze berechnen
   -> Nutzerfrage vektorisieren
   -> ähnlichste Absätze suchen
   -> Top-k-Kontext an gemma3:12b geben
   -> Antwort erzeugen
   -> Antwort + Fundstellen in Streamlit ausgeben
```

---

## Beispielhafter End-to-End-Ablauf

### 1. Nutzer gibt eine URL ein

Zum Beispiel:

```text
https://www.bundesverfassungsgericht.de/SharedDocs/Entscheidungen/DE/2025/03/rs20250326_2bvr150520.html
```

### 2. Nutzer stellt eine Frage

Zum Beispiel:

```text
Unter welchen Voraussetzungen darf eine Ergänzungsabgabe nach Art. 106 Abs. 1 Nr. 6 GG erhoben werden?
```

### 3. Das System verarbeitet das Urteil

* Die Entscheidungsgründe werden geparst.
* Jede Randnummer wird mit ihrem Absatztext gespeichert.
* Für jeden Absatz wird ein Embedding erzeugt.

### 4. Retrieval

Die Frage wird ebenfalls eingebettet.
Dann werden die ähnlichsten Randnummern aus dem Urteil gesucht.

### 5. LLM-Antwort

Die fünf besten Fundstellen werden in einen Prompt eingebaut und an `gemma3:12b` geschickt.

### 6. Ausgabe

Die App zeigt zuerst die Antwort und danach die einzelnen Fundstellen mit:

* Randnummer
* Score
* Text

---

## Vorteile dieser Struktur

Die modulare Struktur hat mehrere Vorteile:

### Bessere Wartbarkeit

Wenn sich das HTML der BVerfG-Seite ändert, muss nur `scraper.py` angepasst werden.

### Bessere Erweiterbarkeit

Neue Features lassen sich leicht ergänzen, zum Beispiel:

* mehrere Urteile gleichzeitig,
* Speicherung von Indizes,
* Reranking,
* Chat-Historie,
* Exportfunktionen.

### Bessere Testbarkeit

Jede Datei hat eine klar abgegrenzte Aufgabe.
Dadurch lassen sich einzelne Funktionen separat testen.

### Trennung von UI und Logik

`app.py` bleibt schlank und verständlich, weil dort nur noch die Streamlit-Steuerung liegt.

---

## Installation

### Python-Abhängigkeiten

```bash
uv add streamlit pandas requests beautifulsoup4 ollama
```

### Ollama-Modelle laden

```bash
ollama pull qwen3-embedding:8b
ollama pull gemma3:12b
```

---

## Start der Anwendung

```bash
uv run streamlit run app.py
```

Danach öffnet sich die Streamlit-App im Browser.

---

## Mögliche Erweiterungen

Das aktuelle System ist bereits ein funktionierender Einstieg in ein juristisches RAG. Sinnvolle nächste Schritte wären:

### 1. Persistenz

Embeddings pro URL speichern, damit sie nicht bei jedem Neustart neu berechnet werden müssen.

### 2. Unterstützung mehrerer Entscheidungen

Mehrere Urteile in einem gemeinsamen Korpus speichern. Dafür könnte die Website des BVerfG systematisch gescraped werden.

### 3. Metadaten erweitern

Zusätzliche DataFrame-Spalten wie:

* `URL`
* `Titel`
* `Datum`
* `Aktenzeichen`
* `ECLI`

### 4. Reranking

Nach dem ersten Embedding-Retrieval noch ein zweites Ranking durchführen.

### 5. PDF-Fallback

Falls HTML nicht sauber geparst werden kann, könnte zusätzlich ein PDF-Fallback integriert werden.
(bei BVerfG-Urteilen aber eher nicht notwendig)