# RAG Q&A

**Note:** See [system_architecture.md](system_architecture.md) for in-detail information about the system design.

### Run guide

1. Build the docker image of API server:
```
make build
```

2. Build the docker image of ChromaDB server:
```
make chromadb
```

3. Run the API Server, docker networking and Start streamlit:
```
make run
```

4. Build and run flask app and UI:
```
make all
```

5. Access the Chatbot UI via `http://localhost:8501/` or
6. Access the API at `POST http://0.0.0.0:5000/answer`. **Example**: 
```
curl -X POST -H 'Accept: */*' -H 'Accept-Encoding: gzip, deflate' -H 'Connection: keep-alive' -H 'Content-Length: 62' -H 'Content-Type: application/x-www-form-urlencoded' -H 'User-Agent: python-requests/2.31.0' -d prompt=Can+you+give+some+summary+about+the+generative+agent%3F http://0.0.0.0:5000/answer
```

7. Some unit testing:
```
$ cd tests
$ pytest -vv --ignore chroma
```