FROM python:3.10-slim

WORKDIR /app

RUN apt-get update -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN apt-get install -y wget
RUN wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb
RUN rm google-chrome-stable_current_amd64.deb

COPY ./requirements.txt .
RUN python -m pip install -r ./requirements.txt

COPY . .

ENTRYPOINT [ "/bin/sh", "/app/entry.sh" ]
