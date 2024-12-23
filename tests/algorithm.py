
### A script to test the algorithm. ###


import os
import pathlib
# Get the current file's directory
current_dir = pathlib.Path(__file__).parent

GEMMA_PATH = current_dir.parent / "models" / "bge-reranker-v2-gemma"
#GEMMA_PATH = current_dir.parent / "models" / "bge-reranker-v2.5-gemma2-lightweight"
M3_PATH = current_dir.parent / "models" / "bge-reranker-v2-m3"

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#from FlagEmbedding import FlagReranker  #replaced with optimised reranker
from FlagEmbedding import FlagLLMReranker
#from FlagEmbedding import LightWeightFlagLLMReranker

import numpy as np



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

llm_reranker = FlagLLMReranker(GEMMA_PATH, normalize=True)
"""llm_reranker = LightWeightFlagLLMReranker(GEMMA_PATH, 
                                          use_fp16=True, 
                                          normalize=True,
                                          cutoff_layers=[28], 
                                          compress_ratio=2, 
                                          compress_layer=[24, 40]
                                          ) # Setting use_fp16 to True speeds up computation with a slight performance degradation"""  #not using because scores were screwy

m3_reranker = OptimizedFlagReranker(M3_PATH, normalize=True)



def rerank_results(query, passages, use_llm=False):
    pairs = [[query, passage] for passage in passages]
    
    reranker = None
    if use_llm:
        reranker = llm_reranker 
    else:
        reranker = m3_reranker
    
    scores = reranker.compute_score(pairs)
    
    
    return scores




if __name__ == "__main__":
    #a quick test script

    query = "What is the capital of France?"
    passages = [
        "Paris is the capital of France.",
        "London is the capital of England.",
        "Berlin is the capital of Germany."
    ]
    import time

    # Pre-code: Start the timer
    start_time = time.time()

    # The line of code you want to measure
    scores = rerank_results(query, passages, use_llm=True)

    # Post-code: Stop the timer and calculate elapsed time in milliseconds
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    # Print the execution time in milliseconds
    print(f"Execution time: {elapsed_time_ms:.2f} ms")

    ## Print results with passages for clarity
    for score, passage in zip(scores, passages):
        print(f"Score: {score:.4f} - {passage}")
