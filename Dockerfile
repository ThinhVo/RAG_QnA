FROM python:3.11

EXPOSE 5000

WORKDIR /app

RUN apt-get update && \
    apt-get install -y nginx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV FLASK_APP=flask_app
ENV FLASK_RUN_HOST=127.0.0.1

RUN chmod +x start_chromadb.sh

CMD [ "flask", "run", "--host=0.0.0.0" ]
