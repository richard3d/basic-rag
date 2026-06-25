from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from query import query_rag

app = FastAPI()

class QueryRequest(BaseModel):
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

class QueryResponse(BaseModel):
    response: str

    @field_validator("response")
    @classmethod
    def response_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("response must not be empty")
        return v

@app.post("/query-rag", response_model=QueryResponse)
def query_rag_endpoint(body: QueryRequest) -> QueryResponse:
    answer = query_rag(body.prompt)
    return QueryResponse(response=answer)
