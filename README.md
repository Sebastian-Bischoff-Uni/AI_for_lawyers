# KI für Juristen

Dieses Repository enthält Materialien und Beispielanwendungen für die Blockveranstaltung „KI für Juristen“ an der juristischen Fakultät der Julius-Maximilians-Universität Würzburg. Der Schwerpunkt liegt auf einer praxisnahen Einführung in Python, Impementation von Large Language Models in Code, Berechnung semantischer Ähnlichkeit von Textsequenzen, Retrieval-Augmented Generation (RAG) sowie der Entwicklung einfacher Legal-Tech-Anwendungen.

## Ziel des Repositories

Das Repository dient als Arbeitsgrundlage für die Lehrveranstaltung und umfasst:

- eine Einführung in Python,
- die Nutzung von Large Language Models (LLMs) lokal und über die API,
- Grundlagen zur semantischen Ähnlichkeitssuche,
- Grundlagen von Retrieval-Augmented Generation (RAG),
- exemplarische Anwendungen im juristischen Kontext.

Die Inhalte sind primär für Lehr- und Lernzwecke konzipiert.

## Aufbau des Repositories

### 01_python_basics

Einführung in die Programmiersprache Python anhand von Jupyter-Notebooks:

- Datentypen
- Methoden
- Operatoren
- Variablen
- Kontrollstrukturen (`if`, `else`, `elif`)
- Schleifen
- Listen
- Funktionen
- Python-Bibliotheken
- objektorientierte Programmierung

Diese Einheit richtet sich insbesondere an Einsteiger ohne Programmiererfahrung.

### 02_llm_call

Notebooks zur Nutzung von Large Language Models:

- `LLMs_API.ipynb`: Nutzung über externe APIs
- `LLMs_local.ipynb`: lokales Hosting von Modellen

### 03_semantic_similarity

Materialien zur Ermittlung der semantischen Ähnlichkeit von Texten, die ein häufigen Abfragemechanismus in RAG-Systemen darstellt:

- semantische Suche in juristischen Texten,
- Erstellung und Nutzung von Embeddings,
- Beispiel anhand von Entscheidungen des Bundesverfassungsgerichts.

### 04_bverfg_rag

Beispiel einer lokalen Anwendung zur Verarbeitung juristischer Texte. Enthält u. a.:

- Konfigurationslogik
- Embedding-Generierung
- Retrieval-Mechanismen
- RAG-Pipeline
- Web-Scraping-Komponenten

### 05_bverfg_app_local

Lokal über Ollama laufende Variante der Anwendung aus `04_bverfg_rag` kombiniert mit einer Streamlit-Benutzeroberfläche.
Eine separate README in diesem Ordner enthält weiterführende Hinweise.

### 06_bverfg_app_gemini

Über Google Gemini laufende Variante der Anwendung aus `04_bverfg_rag` kombiniert mit einer Streamlit-Benutzeroberfläche.

### 07_bverfg_app_openai

Über OpenAI-Modelle laufende Variante der Anwendung aus `04_bverfg_rag` kombiniert mit einer Streamlit-Benutzeroberfläche.

## Projektdateien

- `pyproject.toml`: Projektdefinition und "dependencies"
- `uv.lock`: Lock-Datei für reproduzierbare Installationen
- `.python-version`: Python-Version
- `.env`: Umgebungsvariablen (z. B. API-Schlüssel)
- `.gitignore`: Auflistung von Dateien, die beim Upload auf Github ignoriert werden
- `LICENSE`: Lizenz des Repositories

## Voraussetzungen

- Python (Version gemäß `.python-version`)
- `uv` als Paket- und Environment-Manager

Je nach Anwendungsfall:
- API-Key (für LLM-Zugriffe)
- lokal installierte Modelle: Installation von Ollama auf für [Windows](https://ollama.com/download/windows) oder [Mac](https://ollama.com/download/mac)
