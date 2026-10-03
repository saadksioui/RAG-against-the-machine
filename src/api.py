from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from src.__main__ import RagCLI
import uvicorn


class QueryRequest(BaseModel):
    query: str
    k: int = 5
    search_mode: str = "bm25"


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_cli = RagCLI()
    app.state.service = rag_cli
    yield


app = FastAPI(title="RAG Against The Machine API", lifespan=lifespan)


@app.post('/search')
def rag_search(request: QueryRequest):
    try:
        result = app.state.service.search(request.query, request.k, request.search_mode)
        return {"query": request.query, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/answer')
def rag_answer(request: QueryRequest):
    try:
        result = app.state.service.answer(request.query, request.k, request.search_mode)
        return {"query": request.query, "results": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)