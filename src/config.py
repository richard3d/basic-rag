import os

FILE_OBSERVER_DIRECTORY = os.environ.get("FILE_OBSERVER_DIRECTORY", "/app/docs")
DB_CONNECTION_STR =  os.environ.get("DB_CONNECTION_STR", "postgresql://postgres:password@db:5432/tai-stack")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# Defaults to 768 to match dimensions for the default model nomic-embed-text
EMBEDDING_MODEL_VEC_DIMENSION =os.environ.get("EMBEDDING_MODEL_VEC_DIMENSION", 768)
