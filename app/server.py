#server stuff
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Optional


#reranker stuff
from FlagEmbedding import FlagReranker  #replaced with optimised reranker
from FlagEmbedding import FlagLLMReranker

import os
import pathlib

# Get the current file's directory
current_dir = pathlib.Path(__file__).parent

GEMMA_PATH = current_dir.parent / "models" / "bge-reranker-v2-gemma"
M3_PATH = current_dir.parent / "models" / "bge-reranker-v2-m3"

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Set up the app and load the models
m3_reranker = None
llm_reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    #load the models
    global m3_reranker
    global llm_reranker

    m3_reranker = FlagReranker(M3_PATH, normalize=True)
    llm_reranker = FlagLLMReranker(GEMMA_PATH, normalize=True)  # Initialize the LLM reranker


    #anyhting after the yield call will be ran on shutdown...seems to fail without the yield?
    yield
    pass

app = FastAPI(lifespan=lifespan)


def rerank_results(query, passages, use_llm=False):
    pairs = [[query, passage] for passage in passages]

    reranker = None
    if use_llm:
        reranker = llm_reranker
    else:
        reranker = m3_reranker
    
    scores = reranker.compute_score(pairs)
    
    return scores

class RerankRequest(BaseModel):
    query: str
    passages: List[str]
    use_llm: Optional[bool] = False


@app.post("/rerank")
async def rerank(request: RerankRequest):
    
    print(f"Received query: {request.query}")
    print(f"Received passages: {request.passages}")   
    print(f"Received use_llm: {request.use_llm}")

    return {"response": rerank_results(request.query, request.passages, request.use_llm)}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)