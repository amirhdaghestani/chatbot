"""This module contains necessary configs for chatbot"""
import os


class ChatBotConfig:
    """Necessary configs for chatbot.

    Attributes:
        api_key [required] (str): OpenAI api key.
        chat_engine (str): Chat engine to be used [default: gpt-3.5-turbo].
        max_tokens (int): Maximum number of tokens accepted [default: 4096].
        number_responses (int): Number of returned responses [default: 1].
        stop_by (str): String to stop response by [default: None].
        temperature (float): Determines randomness of model [default: 0.5].
        bot_description (str): Bot description which determines bot personality.

    """
    api_key = str(os.getenv("OPENAI_API_KEY")) \
              if os.getenv("OPENAI_API_KEY") else None
    chat_engine = str(os.getenv("CHAT_ENGINE")) \
                  if os.getenv("CHAT_ENGINE") else "gpt-3.5-turbo"
    max_tokens = int(os.getenv("MAX_TOKENS")) \
                 if os.getenv("MAX_TOKENS") else 512
    num_responses = int(os.getenv("NUM_RESPONSES")) \
                    if os.getenv("NUM_RESPONSES") else 1
    stop_by = str(os.getenv("STOP_BY")) \
              if os.getenv("STOP_BY") else None
    temperature = float(os.getenv("TEMPERATURE")) \
                  if os.getenv("TEMPERATURE") else 0.1
    bot_description = str(os.getenv("BOT_DESC")) \
                      if os.getenv("BOT_DESC") else ""
    delim_context = str(os.getenv("DELIM_CONTEXT")) \
                    if os.getenv("DELIM_CONTEXT") else "\n\n###\n\n"
    prefix_prompt = str(os.getenv("PREFIX_PROMPT")) \
                    if os.getenv("PREFIX_PROMPT") else "Customer: "
    suffix_prompt = str(os.getenv("PREFIX_PROMPT")) \
                    if os.getenv("PREFIX_PROMPT") else "\nAgent:"
