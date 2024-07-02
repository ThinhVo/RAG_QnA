import pytest
import sys
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import load_evaluator

sys.path.append('..')

from flask_app import llm_completion


def test_llm_answer():
    """
    Given temperature=0, the response for the same answer should not have high cosine distance.
    This is to test if the RAG chain is working properly
    """
    prompt = 'Can you give some summary about the generative agent?'
    
    answer = llm_completion(prompt, host="localhost", port=8000)

    gt_answer = "Generative agents are believable simulacra of human behavior that are dynamically conditioned on agents’ changing experiences and environment. They are enabled by a novel architecture that makes it possible for generative agents to remember, retrieve, reflect, interact with other agents, and plan through dynamically evolving circumstances.  The architecture leverages the powerful prompting capabilities of large language models and supplements those capabilities to support longer-term agent coherence, the ability to manage dynamically-evolving memory, and recursively produce more generations."

    evaluator = load_evaluator("pairwise_embedding_distance", distance_metric=EmbeddingDistance.COSINE)
    cosine_distance = evaluator.evaluate_string_pairs(prediction=answer, prediction_b=gt_answer)
    # print(answer)
    assert cosine_distance['score'] <= 0.1
