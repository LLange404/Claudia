import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel       
from typing import List, Optional
import pandas as pd

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
import lancedb



# 1. Konfiguration & Umgebungsvariablen
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("API_KEY") 

# 2. Datenbank-Setup
DB_PATH = "./data"
TABLE_NAME = "judgments"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print(f"Lade Embedding-Modell: {EMBEDDING_MODEL_NAME}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print(f"Verbinde mit LanceDB unter {DB_PATH}...")
db = lancedb.connect(DB_PATH)


# 3. Vektordatenbank aus Data Öffnen

try:
    table = db.open_table(TABLE_NAME)
except Exception as e:
    print(f"Fehler: Tabelle '{TABLE_NAME}' konnte nicht geöffnet werden. Hast du ingest.py ausgeführt? {e}")
    table = None

# 3. LLM-Setup (Claude)
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=20000,
    thinking={
        "type": "enabled",
        "budget_tokens": 4000
    }
)

# 4. FastAPI Setup
app = FastAPI(title="Claudia - Datenbank")

# CORS-Middleware hinzufügen, um Cross-Origin-Anfragen vom Frontend zu erlauben
# Löschen wenn Fertig!

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Erlaubt Anfragen von allen Origins (für Entwicklung)
    allow_credentials=True,
    allow_methods=["*"],  # Erlaubt alle HTTP-Methoden
    allow_headers=["*"],  # Erlaubt alle HTTP-Header
)


# Nutzer schickt Query → API sucht Quellen (Sources) → Claude generiert Antwort → API gibt Response zurück

# Nutzeranfrage
class Query(BaseModel):
    text: str

# Struktur für extrahierte Filter
class SearchFilters(BaseModel):
    court_name: Optional[str] = None
    file_number: Optional[str] = None
    date: Optional[str] = None
    search_query: str # Der bereinigte Suchtext für Vektor/FTS

# Hilfsfunktion zur Filter-Extraktion
async def get_search_params(user_query: str) -> SearchFilters:
    # Wir nutzen das gleiche Modell wie für die Hauptantwort, aber ohne thinking-Block
    extraction_llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=1000
    )
    
    prompt = f"""Analysiere die folgende juristische Suchanfrage und extrahiere Filterkriterien für eine Datenbankabfrage.
    
    WICHTIG: Antworte AUSSCHLIESSLICH mit einem validen JSON-Objekt. Keine Einleitung, kein Text davor oder danach.
    
    JSON-Felder:
    - court_name: Name des Gerichts (z.B. 'Landgericht Köln', 'Oberlandesgericht Celle'). Wenn kein Gericht genannt wird: null.
    - file_number: Aktenzeichen (z.B. '14 U 19/22'). Wenn kein Aktenzeichen genannt wird: null.
    - date: Datum (z.B. '2022-10-05'). Wenn kein Datum genannt wird: null.
    - search_query: Ein aussagekräftiger Suchbegriff für den Inhalt (z.B. 'Mietrecht', 'Hundebiss'). Wenn die Anfrage nur aus Metadaten besteht, nutze ein leeres Wort.
    
    Nutzeranfrage: "{user_query}"
    
    JSON:"""
    
    try:
        response = extraction_llm.invoke([
            ("system", "Du bist ein präziser JSON-Extraktor für juristische Metadaten. Antworte NUR mit JSON."),
            ("user", prompt)
        ])
        
        import json
        import re
        content = str(response.content)
        
        # Versuche JSON-Block zu finden
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        data = json.loads(content)
        
        # Sicherstellen, dass search_query nie leer ist für die Vektorsuche
        if not data.get("search_query"):
            data["search_query"] = user_query
            
        return SearchFilters(**data)
    except Exception as e:
        print(f"Filter-Extraktion fehlgeschlagen: {e}. Nutze Fallback.")
        return SearchFilters(search_query=user_query)

# Quellenangabe für einzelnes Dokument aus der Vektordatenbank
class Source(BaseModel):
    id: Optional[str]
    court_name: Optional[str]
    file_number: Optional[str]
    date: Optional[str]
    snippet: str

# Antwort der API
class Response(BaseModel):
    answer: str
    sources: List[Source]
    total_count: Optional[int] = None

# API Endpunkt
@app.post("/ask", response_model=Response)
async def ask_claudia(query: Query):
    if table is None:
        raise HTTPException(status_code=500, detail="Vektor-Datenbank nicht initialisiert.")

    try:
        print(f"\n--- Incoming Query: '{query.text}' ---")

        # 1. Analyse & Filter-Extraktion
        params = await get_search_params(query.text)
        print(f"Extrahierte Filter: {params}")

        # 2. SQL-Filter für LanceDB aufbauen
        where_clauses = []
        if params.court_name:
            where_clauses.append(f"metadata.court_name = '{params.court_name}'")
        if params.file_number:
            where_clauses.append(f"metadata.file_number = '{params.file_number}'")
        if params.date:
            where_clauses.append(f"metadata.date = '{params.date}'")

        where_sql = " AND ".join(where_clauses) if where_clauses else None

        # NEU: Gesamtzahl in der DB ermitteln, wenn ein Filter gesetzt ist
        total_count_judgments = None
        if where_sql:
            # Wir holen alle gefilterten Metadaten (nur die Spalte metadata reicht)
            matching_df = table.search().where(where_sql).to_pandas()
            if not matching_df.empty:
                # Wir zählen die einzigartigen Aktenzeichen (file_number)
                total_count_judgments = matching_df["metadata"].apply(lambda x: x.get("file_number")).nunique()
                print(f"Echte Urteile in DB für Filter: {total_count_judgments}")
            else:
                total_count_judgments = 0

        # 3. Retrieval (Hybrid: Vector + FTS + SQL-Filter)
        # Wenn nur nach Metadaten gefiltert wird, erhöhen wir k

        # erhöhen wir k, um mehr Ergebnisse zu zeigen.
        k = 10 if (where_sql and len(params.search_query.split()) < 3) else 4
        
        query_vector = embeddings.embed_query(params.search_query)
        
        print(f"Executing Query with k={k} and WHERE: {where_sql}")

        # Vektorsuche (Semantisch) mit Filter
        try:
            vec_query = table.search(query_vector)
            if where_sql:
                vec_query = vec_query.where(where_sql, prefilter=True)
            vector_results = vec_query.limit(k).to_pandas()
        except Exception as e:
            print(f"Vektorsuche Fehler: {e}")
            vector_results = pd.DataFrame()
        
        # FTS-Suche (Stichwortbasiert) mit Filter
        try:
            fts_query = table.search(params.search_query.lower(), query_type="fts")
            if where_sql:
                fts_query = fts_query.where(where_sql, prefilter=True)
            fts_results = fts_query.limit(k).to_pandas()
        except Exception as e:
            print(f"FTS Suche fehlgeschlagen: {e}")
            fts_results = pd.DataFrame()

        # Ergebnisse kombinieren
        combined_df = pd.DataFrame()
        if not fts_results.empty and not vector_results.empty:
            combined_df = pd.concat([fts_results, vector_results]).drop_duplicates(subset=["text"]).head(k)
        elif not fts_results.empty:
            combined_df = fts_results.head(k)
        elif not vector_results.empty:
            combined_df = vector_results.head(k)
        
        # Sonderfall: Wenn GAR NICHTS gefunden wurde trotz Filter, versuchen wir eine reine Metadaten-Suche
        if combined_df.empty and where_sql:
            print("Keine Treffer mit search_query, versuche reinen Metadaten-Filter...")
            combined_df = table.search().where(where_sql).limit(k).to_pandas()

        print(f"--- Retrieved {len(combined_df)} documents ---")

        print(f"--- Retrieved {len(combined_df)} documents for query: '{query.text}' ---")
        
        # Konvertierung in ein für den Prompt-Aufbau hilfreiches Format
        docs_for_prompt = []
        for i, row in combined_df.iterrows():
            docs_for_prompt.append({
                "page_content": row["text"],
                "metadata": row["metadata"]
            })
            print(f"Document {i+1} Page Content (Snippet):\n{row['text'][:200]}...\n--- End Document {i+1} ---")
        
        # 2. Prompting
        context_text = ""
        for i, doc in enumerate(docs_for_prompt):
            meta = doc["metadata"]
            context_text += f"""
        --- Urteil {i+1} ---
        Gericht: {meta.get('court_name', 'unbekannt')}
        Aktenzeichen: {meta.get('file_number', 'unbekannt')}
        Datum: {meta.get('date', 'unbekannt')}
        Inhalt: {doc['page_content']}
        """
        
        # 2. Prompting
        # Wir geben Claude die Info über die Gesamtzahl mit, falls vorhanden
        count_info = f"\nINFO: In der gesamten Datenbank befinden sich insgesamt {total_count_judgments} verschiedene Urteile, die deine Filterkriterien erfüllen." if total_count_judgments is not None else ""
        
        system_prompt = f"""Du bist 'Claudia', eine juristische KI-Assistentin für eine Urteilsdatenbank mit über 250.000 deutschen Gerichtsurteilen.
        {count_info}

        ## Deine Aufgabe
        Analysiere die Frage des Nutzers und beantworte sie auf Basis der bereitgestellten Urteile. 
        Falls der Nutzer nach der Anzahl der Urteile fragt, nutze die oben genannte INFO-Gesamtzahl.

        ## Regeln
        - Beziehe dich **immer konkret** auf ein oder mehrere Urteile (Gericht, Aktenzeichen, Datum)
        - Zitiere nie aus dem Gedächtnis sondern nur aus dem gegebenen Kontext
        - Wenn kein passendes Urteil im Kontext vorhanden ist, sage dies klar und ehrlich
        - Erkläre kurz **warum** ein Urteil relevant für die Frage ist
        - Antworte auf Deutsch, präzise und juristisch korrekt
        - Spekuliere nicht über Urteile die nicht im Kontext stehen

        ## Antwortstruktur
        1. Direkte Antwort auf die Frage
        2. Relevante Urteile mit Begründung
        3. Ggf. Hinweis auf Einschränkungen oder fehlende Treffer
        """
        user_prompt = f"""## Gefundene Urteile aus der Datenbank: {context_text}

        ## Frage des Nutzers: {query.text}

        ## Aufgabe:
        Beantworte die Frage ausschließlich auf Basis der oben genannten Urteile.
        Nenne für jedes relevante Urteil: Gericht, Aktenzeichen, Datum und warum es zur Frage passt.
        Falls kein Urteil passt, teile dies dem Nutzer mit.

        Antwort:"""
        
        # 3. Generation
        messages = [("system", system_prompt), ("user", user_prompt)]
        ai_response = llm.invoke(messages)
        
        # Die Antwort von Anthropic mit "thinking" ist eine Liste von Blöcken.
        # Text aus dem entsprechenden Block extrahieren.
        answer_text = ""
        if isinstance(ai_response.content, list):
            for block in ai_response.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    answer_text += block.get("text", "")
        else:
            answer_text = str(ai_response.content) # Fallback für ältere Formate

        # 4. Quellen aufbereiten
        sources = [] 
        for i, doc in enumerate(docs_for_prompt):
            meta = doc["metadata"]
            sources.append(Source(
                id=str(meta.get("id", "")),
                court_name=meta.get("court_name", "Unbekannt"),
                file_number=meta.get("file_number", "Unbekannt"),
                date=meta.get("date", "Unbekannt"),
                snippet=doc["page_content"][:200] + "..."
            ))
            print("----------------------------")
            
        return Response(answer=answer_text, sources=sources, total_count=total_count_judgments)
        
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Frontend ausliefern
@app.get("/")
async def read_index():
    return FileResponse('web/index.html')

app.mount("/", StaticFiles(directory="web"), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
