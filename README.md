# Basic RAG

A self-contained Retrieval-Augmented Generation (RAG) system built with Python and LangChain. This project was written as a hands-on exercise to understand the core mechanics of RAG pipelines — from document ingestion to vector search to LLM-generated answers — all running locally with no external API keys required.

## What It Does

Drop a text file into a watched directory and the system automatically chunks it, embeds it, and stores the vectors in a PostgreSQL database. Then ask questions against those documents via a REST API and get answers grounded in the ingested content.

```
Document → Chunker → Embedder → pgvector DB
                                     ↑
Question → Embedder → Similarity Search → LLM → Answer
```

## Stack

| Layer | Technology |
|---|---|
| LLM & Embeddings | [Ollama](https://ollama.com) (`llama3.2` + `nomic-embed-text`) |
| Vector Store | [pgvector](https://github.com/pgvector/pgvector) (PostgreSQL extension) |
| RAG Orchestration | [LangChain](https://python.langchain.com) |
| API Server | [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) |
| File Watching | [Watchdog](https://github.com/gorakhargosh/watchdog) |
| Runtime | Python 3.14, [uv](https://github.com/astral-sh/uv) |
| Infrastructure | Docker Compose |

## Architecture

The RAG service (`src/`) has two concurrent responsibilities, run in separate threads from `main.py`:

**Ingest pipeline** — a Watchdog file observer monitors a directory for new files. When a file arrives:
1. `chunk.py` reads it asynchronously and splits it into overlapping text chunks using LangChain's `RecursiveCharacterTextSplitter`
2. `embed.py` calls Ollama (`nomic-embed-text`) to generate a 768-dimensional embedding for each chunk and writes it to pgvector

**Query pipeline** — a FastAPI server exposes a `POST /query-rag` endpoint. When a question arrives:
1. The question is embedded using the same model
2. pgvector finds the top-3 most similar chunks using cosine distance (`<=>` operator)
3. Those chunks are injected into a prompt template and sent to `llama3.2` via LangChain's `ChatOllama`
4. The grounded answer is returned in the response

## Project Structure

```
basic-rag/
├── src/
│   ├── main.py       # Entry point — starts API server and file observer concurrently
│   ├── server.py     # FastAPI app with /query-rag endpoint
│   ├── ingest.py     # Watchdog observer that triggers the embed pipeline on new files
│   ├── chunk.py      # Async streaming text chunker (LangChain splitter)
│   ├── embed.py      # Generates embeddings and writes to pgvector
│   ├── query.py      # Retrieves relevant chunks and calls the LLM
│   └── config.py     # Environment-based configuration
├── pgvector/
│   └── init-scripts/init.sql   # Enables the pgvector extension on startup
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml
```

## Running It

**Prerequisites:** Docker and Docker Compose.

```bash
# Pull images, build the RAG container, and start everything
docker compose up --build
```

On first run, `ollama-init` pulls `llama3.2` and `nomic-embed-text` — this may take a few minutes depending on your connection.

Services:
- RAG API: `http://localhost:8000`
- Ollama: `http://localhost:11434`

## Usage

**Ingest a document** — copy any `.txt` file into `./docs/` (created automatically on first run):

```bash
echo "The capital of France is Paris." > ./docs/facts.txt
```

The file observer picks it up immediately and logs the processing steps.

**Query the RAG system:**

```bash
curl -X POST http://localhost:8000/query-rag \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?"}'
```

```json
{"response": "The capital of France is Paris."}
```

## GPU Acceleration

CPU inference works out of the box. For GPU support on Linux with an NVIDIA card, uncomment the `deploy` block in `docker-compose.yaml` under the `ollama` service and install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

> **Note:** Docker on Apple Silicon cannot access the GPU — CPU inference is used automatically on macOS.

## Key Concepts Explored

- **Chunking strategy** — overlapping windows (`chunk_size=1000`, `chunk_overlap=100`) prevent context from being cut off at chunk boundaries
- **Embedding model separation** — using a dedicated embedding model (`nomic-embed-text`) separate from the generative model (`llama3.2`) is standard practice; each is optimized for its task
- **Vector similarity search** — pgvector's `<=>` cosine distance operator finds semantically similar chunks without a full table scan
- **Prompt grounding** — the system prompt instructs the LLM to answer strictly from the retrieved context and admit when it cannot find the answer, reducing hallucination
- **Re-ingestion** — dropping a file that was previously ingested deletes the old embeddings before re-processing, preventing duplicate chunks
- **Streaming chunker** — `chunk.py` reads files in blocks and yields chunks incrementally rather than loading the whole file into memory

## Design Trade-offs & Potential Improvements

**Add an LLM gateway (e.g. LiteLLM) between the RAG service and Ollama**

Currently, `query.py` and `embed.py` call Ollama directly using `langchain_ollama`'s `ChatOllama` and `OllamaEmbeddings`. This works, but it couples the application code to Ollama as a provider. Swapping to a different model or backend requires touching the RAG code itself.

A cleaner approach is to introduce a gateway like [LiteLLM](https://github.com/BerriAI/litellm) between the RAG service and Ollama. LiteLLM exposes an OpenAI-compatible REST API in front of any backend (Ollama, OpenAI, Anthropic, Bedrock, etc.), so the RAG service can speak a single standard interface regardless of what's running underneath:

```python
# Instead of this (coupled to Ollama)
from langchain_ollama import ChatOllama, OllamaEmbeddings
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama3.2")

# Do this (coupled to an OpenAI-compatible interface)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
llm = ChatOpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY, model="llama3.2")
```

The benefits:

- **Provider portability** — switching from a local Ollama model to a cloud provider becomes a configuration change in the gateway, not a code change in the RAG service
- **Centralized model routing** — all model traffic flows through one place, making it straightforward to add logging, rate limiting, cost tracking, or fallback models
- **Consistent interface** — the OpenAI-compatible API is now the de facto standard for LLM interoperability; building against it future-proofs the integration

**Use LangChain's `PGVector` store instead of raw SQL**

`embed.py` and `query.py` manage the database directly — DDL statements to create the table, raw `psycopg` calls to insert embeddings, and a hand-written cosine similarity query using pgvector's `<=>` operator. This was intentional for learning purposes: writing the SQL by hand makes the mechanics of vector search visible.

In a production system, this would be replaced by LangChain's [`PGVector`](https://python.langchain.com/docs/integrations/vectorstores/pgvector/) vector store (already a dependency in `pyproject.toml` as `langchain-postgres`). It handles schema creation, upserts, and similarity search internally, reducing the database layer to a few lines:

```python
from langchain_postgres.vectorstores import PGVector

store = PGVector(embeddings=embedder, connection=DB_CONNECTION_STR, collection_name="docs")
store.add_texts(chunks)                          # replaces embed.py's insert loop
results = store.similarity_search(question, k=3) # replaces query.py's raw SQL
```

**Add an ORM layer**

Related to the above: the direct `psycopg` calls in `embed.py` mix connection management, schema definition, and business logic in one place. Introducing an ORM like [SQLAlchemy](https://www.sqlalchemy.org/) (which `langchain-postgres` already uses internally) would separate these concerns — models describe the schema, sessions manage transactions, and the application code never writes SQL strings. This also makes it easier to add new tables (e.g. ingestion history, per-document metadata) without the schema management spreading across multiple files.
