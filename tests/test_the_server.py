import requests

def call_llm_service(query, passages):#, use_llm=False):
    response = requests.post(
        "http://localhost:8000/rerank",
        json={"query": query, "passages": passages}#, "use_llm": use_llm}
    )
    return response.json()["response"]

if __name__ == "__main__":
    #a quick test script

    query = "What is the capital of France?"
    passages = [
        "London is the capital of England.",
        "Paris is the capital of France.",
        "Berlin is the capital of Germany."
    ]
    import time

    # Pre-code: Start the timer
    start_time = time.time()

    # The line of code you want to measure
    scores = call_llm_service(query, passages)#, use_llm=False)

    # Post-code: Stop the timer and calculate elapsed time in milliseconds
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000

    # Print the execution time in milliseconds
    print(f"Execution time: {elapsed_time_ms:.2f} ms")

    print(scores)
    ## Print results with passages for clarity
    for score, passage in zip(scores, passages):
        print(f"Score: {score:.4f} - {passage}")