FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -y

COPY ./requirements.txt .
RUN python -m pip install -r ./requirements.txt

COPY . .

ENTRYPOINT [ "/bin/sh", "/app/entry.sh" ]
