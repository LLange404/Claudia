import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel       
from typing import List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import LanceDB
from langchain_community.embeddings import HuggingFaceEmbeddings
import lancedb



# 1. Konfiguration & Umgebungsvariablen
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("API_KEY") 

# 2. Datenbank-Setup
DB_PATH = "./data"
TABLE_NAME = "judgments"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print(f"Lade Embedding-Modell: {EMBEDDING_MODEL_NAME}...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print(f"Verbinde mit LanceDB unter {DB_PATH}...")
db = lancedb.connect(DB_PATH)


# 3. Vektordatenbank aus Data Öffnen + mit Langchain verbinden

try:
    table = db.open_table(TABLE_NAME)
    vectorstore = LanceDB(connection=db, table_name=TABLE_NAME, embedding=embeddings)
except Exception as e:
    print(f"Fehler: Tabelle '{TABLE_NAME}' konnte nicht geöffnet werden. Hast du ingest.py ausgeführt? {e}")
    vectorstore = None

# 3. LLM-Setup (Claude)
llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    anthropic_api_key=ANTHROPIC_API_KEY,
    max_tokens=20000,
    model_kwargs={
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4000
        }
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

# API Endpunkt
@app.post("/ask", response_model=Response)
async def ask_claudia(query: Query):
    if not vectorstore:
        raise HTTPException(status_code=500, detail="Vektor-Datenbank nicht initialisiert.")
    
    try:
        print(f"\n--- Incoming Query: '{query.text}' ---")
        
        # 1. Retrieval
        docs = vectorstore.similarity_search(query.text, k=4) # k ist die Anzahl der Urteile die der Suchanfrage am ähnlichsten sind.

        print(f"--- Retrieved {len(docs)} documents for query: '{query.text}' ---")
        for i, d in enumerate(docs):
            print(f"Document {i+1} Page Content (Full):\n{d.page_content}\n--- End Document {i+1} ---")
        
        
        # So kann Claude die Metadaten direkt in der Antwort verwenden, ohne sie selbst aus einem Fließtext heraussuchen zu müssen.
        
        context_text = ""
        for i, doc in enumerate(docs):
            context_text += f"""
        --- Urteil {i+1} ---
        Gericht: {doc.metadata.get('court_name', 'unbekannt')}
        Aktenzeichen: {doc.metadata.get('file_number', 'unbekannt')}
        Datum: {doc.metadata.get('date', 'unbekannt')}
        Inhalt: {doc.page_content}
        """
        
        # 2. Prompting
        system_prompt = """Du bist 'Claudia', eine juristische KI-Assistentin für eine Urteilsdatenbank mit über 250.000 deutschen Gerichtsurteilen.

        ## Deine Aufgabe
        Analysiere die Frage des Nutzers und beantworte sie ausschließlich auf Basis der bereitgestellten Urteile aus der Datenbank.

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
        sources = [] # Initialize as empty list
        for i, d in enumerate(docs):
            # print(f"--- Debugging Source {i+1} ---")
            # print(f"Document ID: {d.metadata.get('id', 'N/A')}")
            # print(f"Document Metadata: {d.metadata}")
            # print(f"Court Name (from metadata): {d.metadata.get('court_name', 'Default Unbekannt')}")
            # print(f"File Number (from metadata): {d.metadata.get('file_number', 'Default Unbekannt')}")
            # print(f"Date (from metadata): {d.metadata.get('date', 'Default Unbekannt')}")
            # print(f"Snippet: {d.page_content[:200]}...")

            sources.append(Source(
                id=str(d.metadata.get("id", "")),
                court_name=d.metadata.get("court_name", "Unbekannt"),
                file_number=d.metadata.get("file_number", "Unbekannt"),
                date=d.metadata.get("date", "Unbekannt"),
                snippet=d.page_content[:200] + "..."
            ))
            print("----------------------------")
            
        return Response(answer=answer_text, sources=sources)
        
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
