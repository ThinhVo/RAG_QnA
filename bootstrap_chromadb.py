import os
import chromadb
from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

chroma_collection_name = "generative_agent"

def embed_documents():
    """
    Embedding the base knowledge document for information retrieval
    Base knowledge is parsed and saved in `data/generative_agent.txt`
    """
    pages = []

    with open(os.path.join('data', 'generative_agent.txt'), 'r') as f:
        texts = f.read()

    pages = texts.split('\n ---- \n')
    # chroma_client = chromadb.Client()
    chroma_client = chromadb.HttpClient(host="chroma-server-1")

    embeddings_model = OpenAIEmbeddings()

    embeddings = embeddings_model.embed_documents(pages)
    try:
        collection = chroma_client.create_collection(name=chroma_collection_name)
    except:
        # collection = chroma_client.get_collection(name=chroma_collection_name)
        chroma_client.delete_collection(chroma_collection_name)
        collection = chroma_client.create_collection(name=chroma_collection_name)

    collection.add(
        documents=pages,
        metadatas=[{"source": "generative_agent.txt"}] * len(pages),
        ids=[f'page {i}' for i in range(len(pages))],
        embeddings=embeddings
    )

    print(f'Embedded {len(pages)} pages')



if __name__ == "__main__":
    embed_documents()