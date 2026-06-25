import psycopg
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import DB_CONNECTION_STR, OLLAMA_BASE_URL

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based strictly on the provided context.
If you do not know the answer or if it's not explicitly in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: 
{question}

Answer:"""

def retrieve_chunks(question: str, k: int = 3) -> list[str]:
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
    query_embedding = embedder.embed_query(question)
    vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    with psycopg.connect(DB_CONNECTION_STR) as conn:
        rows = conn.execute(
            """
            SELECT chunk_text
            FROM embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vec_str, k),
        ).fetchall()

    return [row[0] for row in rows]

def query_rag(question: str) -> str:
    chunks = retrieve_chunks(question)
    context = "\n\n".join(chunks)

    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama3.2", temperature=0.2)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": question})
