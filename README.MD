# ⚖️ Claudia - Juristische KI-Assistentin

Claudia ist eine spezialisierte **RAG-Anwendung (Retrieval-Augmented Generation)** zur Analyse und Abfrage einer umfangreichen Datenbank deutscher Gerichtsurteile. Sie kombiniert moderne Sprachmodelle mit effizienter Vektorsuche und klassischer Stichwortsuche, um präzise, quellenbasierte Antworten auf juristische Fragestellungen zu liefern.

![Claudia Avatar](web/claudia_avatar.webp)

## 🚀 Hauptmerkmale

- **Hybride Suche:** Kombination aus semantischer Vektorsuche (versteht Konzepte) und Full-Text Search (FTS) für exakte Begriffstreffer (z.B. "Rauhaardackel" oder "§ 812 BGB").
- **Metadaten-Filterung:** Präzise Einschränkung der Suche nach Gericht, Aktenzeichen oder Datum durch automatisierte LLM-Analyse der Nutzeranfrage.
- **Transparente Quellen:** Jede Antwort enthält direkte Verweise auf die gefundenen Urteile inklusive Aktenzeichen und Gericht.
- **Statistische Einblicke:** Claudia kann die exakte Anzahl der Urteile in der Datenbank bestimmen, die bestimmte Filterkriterien erfüllen.
- **Modernes Interface:** Elegantes Dark-Mode Frontend mit einer Side-by-Side Ansicht von Avatar und Chatfenster.

## 🛠 Tech-Stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Vektordatenbank:** [LanceDB](https://lancedb.com/) (Serverless, performant, unterstützt hybride Suche)
- **LLM:** [Anthropic Claude 3.5 Sonnet](https://www.anthropic.com/claude) (über LangChain)
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768 Dimensionen)
- **Framework:** [LangChain](https://www.langchain.com/) für Orchestrierung und Prompt-Management
- **Frontend:** Vanilla JavaScript, HTML5, CSS3 (Modernes Flexbox/Grid Layout)


## 🔍 Architektur-Details

### RAG-Pipeline
1. **Query-Analyse:** Die Nutzeranfrage wird von einem LLM-Parser analysiert, um Metadaten-Filter (Gericht, Datum, AZ) zu extrahieren.
2. **Retrieval:** 
   - Ein SQL-Filter wird auf die LanceDB angewendet.
   - Eine hybride Suche (Vektor + FTS) findet die `k` relevantesten Textstellen.
   - Falls nach einer Anzahl gefragt wird, ermittelt das System die `nunique` Aktenzeichen in den Metadaten.
3. **Augmentation:** Die gefundenen Textstellen werden zusammen mit der Gesamtzahl und dem System-Prompt an Claude übermittelt.
4. **Generation:** Claude generiert eine präzise Antwort basierend *ausschließlich* auf dem bereitgestellten Kontext.

### Datenbank-Schema
| Feld | Typ | Beschreibung |
| :--- | :--- | :--- |
| `vector` | Float32 Array (768) | Semantischer Vektor (MPNet) |
| `text` | UTF-8 String | Der eigentliche Urteilstext (Chunk) |
| `metadata` | Struct | Enthält `court_name`, `file_number`, `date` |