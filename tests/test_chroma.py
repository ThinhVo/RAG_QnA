import chromadb
import pytest

import os

def test_chromadb():
    """ 
    Test if all documents are successfully added to the `generative_agent` collection
    """
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection_name = "generative_agent"
    added_documents = chroma_client.get_collection(chroma_collection_name).count()

    pages = []

    with open(os.path.join('..', 'data', 'generative_agent.txt'), 'r') as f:
        texts = f.read()

    pages = texts.split('\n ---- \n')
    total_documents = len(pages)

    assert added_documents == total_documents