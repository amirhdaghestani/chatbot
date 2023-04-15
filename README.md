# CHATBOT
A module to provide a chatbot based on OpenAI language models.

# Run
In order to run the chatbot, you can either use docker or run locally.
in both ways, you need to set your OpenAI API key.

## Run Docker
build the docker file inside src folder via:

```bash
cd src
docker build -t chatbot:tag .
```
then, run the created image and expose the specified port:
```bash
docker run -p PORTNUMBER:8051 --env OPENAI_API_KEY=YOUR_OPENAI_API_KEY chatbot:tag
```

## Run locally
Install requirements from requirement.txt file in the src folder via:
```bash
cd src
pip install -r requirements.txt
```
then run the following command:
```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY streamlit run app.py
```