# CHATBOT
A module to provide a chatbot based on OpenAI language models.

# Run
In order to run the chatbot, you can either use docker or run locally.
in both ways, you need to set your OpenAI API key.

## Run Docker Compose
Run docker-compose:

```bash
cd src
docker-compose up --build
```

Note to provide 'elasticsearch.env' and 'chatbot.env' enviroment variables.
Also put necessary resources in the 'resources' folder.