build:
	docker build -t llm-agent:latest -f ./Dockerfile .

chromadb:
	./start_chromadb.sh

run:
	docker run -itd -p 5000:5000 --name llm-agent --network chroma_net -v bootstrap_chromadb.py:/bootstrap_chromadb.py llm-agent:latest
	docker exec -ti llm-agent python bootstrap_chromadb.py
	streamlit run streamlit_app.py

all: build chromadb run
