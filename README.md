# Rerank Server

An implementation of the bge-reanker-v2-gemma LLM based ranker running locally as a server. 

This server can be used to rerank search results or any other list of items based on their relevance to a given query. The downstream LLM can then be instructed to use the rerank scores to assist in its analysis. From experimentation its important not to overstep 'assist' in the instructions and still require the LLM to perform its own analysis and validation else it has the tendency to be lazy and just select the highest ranked items even if there are other potential candidates with similarly high scores.

# Setup
1. clone the repo

2. in the models/ directory clone the model with:
git clone https://huggingface.co/BAAI/bge-reranker-v2-m3

3. build the docker image:  
docker buildx build . -t rerank-server:latest

4. run the server: 
docker run -p 8000:8000 rerank-server


# Usage
The server exposes a single endpoint at /rerank which accepts a POST request with the following JSON payload:   
json={"query": query, "passages": passages}

- `query`: A string representing the search query or any other input text.  
- `passages`: An array of strings where each string is a passage or item to be ranked.

The server will return a JSON response with the following structure:  
[float]
- An array of floats where each float represents the score assigned to the corresponding passage in the input list. floats are normalised between 0 and 1, with higher values indicating better matches.









