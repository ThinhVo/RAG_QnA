# Importing necessary modules: The code imports Flask, render_template, request from Flask, os, and openai modules.
from flask import Flask, render_template, request, Response
import openai
import os
import json
import chromadb

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

chroma_collection_name = "generative_agent"

# """
# Embedding the base knowledge document for information retrieval
# Base knowledge is parsed and saved in `data/generative_agent.txt`
# """
# def embed_documents():
#     pages = []

#     with open(os.path.join('data', 'generative_agent.txt'), 'r') as f:
#         texts = f.read()

#     pages = texts.split('\n ---- \n')
#     chroma_client = chromadb.Client()

#     embeddings_model = OpenAIEmbeddings()

#     embeddings = embeddings_model.embed_documents(pages)
#     try:
#         collection = chroma_client.create_collection(name=chroma_collection_name)
#     except:
#         # collection = chroma_client.get_collection(name=chroma_collection_name)
#         chroma_client.delete_collection(chroma_collection_name)
#         collection = chroma_client.create_collection(name=chroma_collection_name)

#     collection.add(
#         documents=pages,
#         metadatas=[{"source": "generative_agent.txt"}] * len(pages),
#         ids=[f'page {i}' for i in range(len(pages))],
#         embeddings=embeddings
#     )
#     return chroma_client, embeddings_model


def llm_completion(user_prompt, host="chroma-server-1", port=8000):
    chroma_client = chromadb.HttpClient(host=host, port=port)
    embeddings_model = OpenAIEmbeddings()

    vectorstore = Chroma(collection_name=chroma_collection_name, client=chroma_client, embedding_function=embeddings_model)
    fine_tuned_model =  os.getenv('FT_MODEL')

    # Prompt template
    from langchain_core.prompts import PromptTemplate


    template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Use three sentences maximum and keep the answer as detailed as possible.

                {context}

                Question: {question}

                Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name=fine_tuned_model, temperature=0)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    
    print(chroma_client.get_collection(chroma_collection_name).count())


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(user_prompt)


# Creating an instance of the Flask class: The Flask() constructor is used to create an instance of the Flask class and assign it to the app variable.
# Setting OpenAI API key: The OpenAI API key is set using os.getenv() function that retrieves the value of the environment variable "OPENAI_API_KEY".
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Defining routes: The app has three routes: '/' (the default route), '/answer'.
# Index route: The index route ('/') renders a template called "index.html" using the render_template() function.
@app.route("/", methods=["GET"])
def index():
    return '<p>Use route `answer` to invoke the LLM model</p>'

# Answer route: The '/answer' route is used to generate a response based on user input. 
# It takes input values from the web form that are passed to the function via the HTTP POST method. 
# The function then uses OpenAI's API to generate a response using the GPT-3 model with the specified prompt and topic. 
@app.route("/answer", methods=["POST"])
def answer():
    print('----' , request.form.to_dict())
    # topic = request.form["topic"]
    prompt = request.form.to_dict()["prompt"]
    # model = os.environ['FT_MODEL']
    # completions = openai.Completion.create(engine=model, prompt=prompt + " " + topic, max_tokens=1024, n=1,stop=None,temperature=0.7)
    # message = completions.choices[0].text
    message = {'completion': llm_completion(user_prompt=prompt)}
    return Response(response=json.dumps(message), status=200, mimetype="application/json")
    # return message

# Running the app: The if __name__ == "__main__": statement ensures that the app is only run if the script is executed directly, and not if it is imported as a module. 
# The app.run() function starts the Flask development server on the local host and enables debug mode.
if __name__ == "__main__":
    app.run(debug=True)