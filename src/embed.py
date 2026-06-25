import psycopg
from langchain_ollama import OllamaEmbeddings
from chunk import chunk_text_file
from config import DB_CONNECTION_STR, OLLAMA_BASE_URL, EMBEDDING_MODEL, EMBEDDING_MODEL_VEC_DIMENSION

async def embed_text_file(file_path):
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    print("Connecting to DB...")
    async with await psycopg.AsyncConnection.connect(DB_CONNECTION_STR) as conn:
        print("DB connection established successfully!")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                file_path TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector({EMBEDDING_MODEL_VEC_DIMENSION}),
                UNIQUE (file_path, chunk_text)
            )
        """)
        # Delete and re-ingest data
        await conn.execute(
                "DELETE FROM embeddings WHERE file_path=%s",
                (file_path,),
            )
        async for chunk in chunk_text_file(file_path):
            embedding = embedder.embed_query(chunk)
            vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
            await conn.execute(
                "INSERT INTO embeddings (file_path, chunk_text, embedding) "
                "VALUES (%s, %s, %s::vector)",
                (file_path, chunk, vec_str),
            )
