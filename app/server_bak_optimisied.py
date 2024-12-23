#unused. contained a slightly faster m3 implementation and the option for an llm gemma2 / 2.5 reranker....not worth the hassle re docker image of longer load times at this stage

#server stuff
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Optional




#reranker stuff
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#from FlagEmbedding import FlagReranker  #replaced with optimised reranker
#from FlagEmbedding import FlagLLMReranker
#from FlagEmbedding import LightWeightFlagLLMReranker

import numpy as np

import os
import pathlib

# Get the current file's directory
current_dir = pathlib.Path(__file__).parent

#GEMMA_PATH = current_dir.parent / "models" / "bge-reranker-v2-gemma"
M3_PATH = current_dir.parent / "models" / "bge-reranker-v2-m3"

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# Set up the app and load the models
#llm_reranker = None
m3_reranker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    #load the models
    global m3_reranker#, llm_reranker
    #llm_reranker = FlagLLMReranker(GEMMA_PATH, normalize=True)
    m3_reranker = OptimizedFlagReranker(M3_PATH, normalize=True)

    #anyhting after the yield call will be ran on shutdown...seems to fail without the yield?
    yield
    pass

app = FastAPI(lifespan=lifespan)





def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

class OptimizedFlagReranker:
    def __init__(self, model_path, normalize=True):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.normalize = normalize
        
    def compute_score(self, pairs):
        # Direct tokenizer call using __call__ method
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**encoded)
            scores = outputs.logits.squeeze(-1)

        if self.normalize:
        # Apply sigmoid function to normalize scores
            scores = [sigmoid(score) for score in scores]
        
        return scores



#adjusted to always use the m3 reranker / remove llm reranker option
def rerank_results(query, passages):#, use_llm=False):
    pairs = [[query, passage] for passage in passages]
    
    """if use_llm:
        reranker = llm_reranker 
    else:
        reranker = m3_reranker"""

    reranker = m3_reranker
    
    scores = reranker.compute_score(pairs)
    
    return scores

class RerankRequest(BaseModel):
    query: str
    passages: List[str]
    #use_llm: Optional[bool] = False



@app.post("/rerank")
async def rerank(request: RerankRequest):
    
    print(f"Received query: {request.query}")
    print(f"Received passages: {request.passages}")
    #print(f"Using LLM for reranking: {request.use_llm}")    

    return {"response": rerank_results(request.query, request.passages)}#, request.use_llm)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)