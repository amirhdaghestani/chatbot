"""This module contains necessary configs for chatbot"""
import os
from enum import Enum

from config.vector_config import EmbeddingModel

class ChatBotModel(Enum):
    GPT4 = "gpt-4"
    TURBO = "gpt-3.5-turbo"
    DAVINCI = "text-davinci-003"
    DAVINCIFAQ = "davinci:ft-personal:faq-9epoch-2023-05-13-17-23-45"
    DAVINCIFAQ4 = "davinci:ft-personal:faq-2023-05-07-05-25-57"
    DAVINCIRBT = "davinci:ft-personal:rbt-25-1100-ca-2023-04-30-12-42-56"
    DAVINCICHAT = "davinci:ft-personal:chat-2023-05-13-20-08-50"


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
    delim_botdesc = str(os.getenv("DELIM_BOTDESC")) \
                    if os.getenv("DELIM_BOTDESC") else "\n\n"
    delim_context = str(os.getenv("DELIM_CONTEXT")) \
                    if os.getenv("DELIM_CONTEXT") else "\n\n###\n\n"
    delim_history = str(os.getenv("DELIM_HISTORY")) \
                    if os.getenv("DELIM_HISTORY") else "\n\n"
    prefix_context = str(os.getenv("PREFIX_CONTEXT")) \
                    if os.getenv("PREFIX_CONTEXT") else "Context:\n"
    prefix_prompt = str(os.getenv("PREFIX_PROMPT")) \
                    if os.getenv("PREFIX_PROMPT") else "Customer: "
    suffix_prompt = str(os.getenv("PREFIX_PROMPT")) \
                    if os.getenv("PREFIX_PROMPT") else "\nAgent:"
    add_context = bool(os.getenv("ADD_CONTEXT")) \
                  if os.getenv("ADD_CONTEXT") else False
    embedding_model = EmbeddingModel(os.getenv("EMBEDDING_MODEL")) \
                      if os.getenv("EMBEDDING_MODEL") else EmbeddingModel.ZIBERT
    max_history = int(os.getenv("MAX_HISTORY")) \
                   if os.getenv("MAX_HISTORY") else 5
    threshold_context = float(os.getenv("THRESHOLD_CONTEXT")) \
                        if os.getenv("THRESHOLD_CONTEXT") else 0.7
    num_retrieve_context = int(os.getenv("NUM_RETRIEVE_CONTEXT")) \
                           if os.getenv("NUM_RETRIEVE_CONTEXT") else 10
    
    def __init__(self, chat_engine: str=None, api_key: str=None,
                 max_tokens: int=None,  num_responses: int=None,
                 stop_by: str=None, temperature: int=None,
                 bot_description: str=None, delim_botdesc: str=None,
                 delim_context: str=None, delim_history: str=None,
                 prefix_context: str=None, prefix_prompt: str=None, 
                 suffix_prompt: str=None, add_context: bool=None, 
                 max_history: int=None, threshold_context: float=None,
                 num_retrieve_context: int=None) -> None:
        """Initializer of class"""
        if chat_engine:
            self.chat_engine = chat_engine
        if api_key:
            self.api_key = api_key
        if max_tokens:
            self.max_tokens = max_tokens
        if num_responses:
            self.num_responses = num_responses
        if stop_by:
            self.stop_by = stop_by
        if temperature:
            self.temperature = temperature
        if bot_description:
            self.bot_description = bot_description
        if delim_botdesc:
            self.delim_botdesc = delim_botdesc
        if delim_context:
            self.delim_context = delim_context
        if prefix_context:
            self.prefix_context = prefix_context
        if prefix_prompt:
            self.prefix_prompt = prefix_prompt
        if suffix_prompt:
            self.suffix_prompt = suffix_prompt
        if add_context:
            self.add_context = add_context
        if max_history:
            self.max_history = max_history
        if delim_history:
            self.delim_history = delim_history
        if threshold_context:
            self.threshold_context = threshold_context
        if num_retrieve_context:
            self.num_retrieve_context = num_retrieve_context

def get_chat_config(chat_engine: ChatBotModel=None, add_context: bool=None,
                    embedding_model: EmbeddingModel=None, max_history: int=None):
    """Function to set chat_config for each chat engine."""
    chatbot_config = ChatBotConfig()

    # Shared configs
    if chat_engine:
        chatbot_config.chat_engine = chat_engine.value

    if add_context is not None:
        chatbot_config.add_context = add_context

    if embedding_model:
        chatbot_config.embedding_model = embedding_model

    chatbot_config.bot_description = (
        "تو ربات هوشمند پشتیبانی شرکت همراه اول هستی که کاربران همراه اول را " \
        "برای یافتن پاسخ سوال‌هایشان راهنمایی می‌کنی. با صداقت کامل به سوال‌ها " \
        "پاسخ بده و اگر نمی‌دانستی بگو متاسفانه پاسخ سوال شما را نمیدانم."
    )

    # Exclusive configs
    if chat_engine == ChatBotModel.GPT4 or chat_engine == ChatBotModel.TURBO:
        chatbot_config.max_tokens = 512
        chatbot_config.num_responses = 1
        chatbot_config.delim_botdesc = ""
        chatbot_config.delim_context = ""
        chatbot_config.delim_context = ""
        chatbot_config.prefix_context = "Context:\n"
        chatbot_config.prefix_prompt = ""
        chatbot_config.suffix_prompt = ""
        chatbot_config.temperature = 0.3
        chatbot_config.max_history = 2
        chatbot_config.threshold_context = 0.7
        chatbot_config.num_retrieve_context = 5
    
    elif chat_engine == ChatBotModel.DAVINCI:
        chatbot_config.max_tokens = 512
        chatbot_config.num_responses = 1
        chatbot_config.delim_botdesc = "\n\n"
        chatbot_config.delim_context = "\n\n###\n\n"
        chatbot_config.delim_context = "\n\n"
        chatbot_config.prefix_context = "Context:\n"
        chatbot_config.prefix_prompt = "Question:\n"
        chatbot_config.suffix_prompt = "\nAnswer:"
        chatbot_config.temperature = 0.3
        chatbot_config.max_history = 2
        chatbot_config.threshold_context = 0.7
        chatbot_config.num_retrieve_context = 1

    elif chat_engine == ChatBotModel.DAVINCIFAQ \
        or chat_engine == ChatBotModel.DAVINCIFAQ4:
        chatbot_config.max_tokens = 512
        chatbot_config.num_responses = 1
        chatbot_config.delim_botdesc = "\n\n"
        chatbot_config.delim_context = "\n\n###\n\n"
        chatbot_config.delim_context = "\n\n"
        chatbot_config.stop_by = "\n###\n"
        chatbot_config.prefix_context = "Context:\n"
        chatbot_config.prefix_prompt = "Customer: "
        chatbot_config.suffix_prompt = "\nAgent:"
        chatbot_config.temperature = 0.1
        chatbot_config.max_history = 2
        chatbot_config.threshold_context = 0.7
        chatbot_config.num_retrieve_context = 1

    elif chat_engine == ChatBotModel.DAVINCIRBT:
        chatbot_config.max_tokens = 512
        chatbot_config.num_responses = 1
        chatbot_config.delim_botdesc = "\n\n"
        chatbot_config.delim_context = "\n\n###\n\n"
        chatbot_config.delim_context = "\n\n"
        chatbot_config.prefix_context = "Context:\n"
        chatbot_config.prefix_prompt = "Customer: "
        chatbot_config.suffix_prompt = "\nAgent: "
        chatbot_config.temperature = 0.1
        chatbot_config.max_history = 2
        chatbot_config.threshold_context = 0.7
        chatbot_config.num_retrieve_context = 1
    
    elif chat_engine == ChatBotModel.DAVINCICHAT:
        chatbot_config.max_tokens = 512
        chatbot_config.num_responses = 1
        chatbot_config.delim_botdesc = "\n\n"
        chatbot_config.delim_context = "\n\n###\n\n"
        chatbot_config.delim_context = "\n\n"
        chatbot_config.stop_by = ["\n###\n", "\n"]
        chatbot_config.prefix_context = "Context:\n"
        chatbot_config.prefix_prompt = "کاربر: "
        chatbot_config.suffix_prompt = "\nربات:"
        chatbot_config.temperature = 0.1
        chatbot_config.max_history = 2
        chatbot_config.threshold_context = 0.7
        chatbot_config.num_retrieve_context = 1

    # To overwrite
    if max_history is not None:
        chatbot_config.max_history = max_history

    # Set Configs and Overwrite if enviroment variable exists.
    cg_to_return = ChatBotConfig()

    if not os.getenv("CHAT_ENGINE"):
        cg_to_return.chat_engine = chatbot_config.chat_engine
    if not os.getenv("MAX_TOKENS"):
        cg_to_return.max_tokens = chatbot_config.max_tokens
    if not os.getenv("NUM_RESPONSES"):
        cg_to_return.num_responses = chatbot_config.num_responses 
    if not os.getenv("BOT_DESCRIPTION"):
        cg_to_return.bot_description = chatbot_config.bot_description
    if not os.getenv("DELIM_BOTDESC"):
        cg_to_return.delim_botdesc = chatbot_config.delim_botdesc
    if not os.getenv("DELIM_CONTEXT"):
        cg_to_return.delim_context = chatbot_config.delim_context
    if not os.getenv("DELIM_HISTORY"):
        cg_to_return.delim_history = chatbot_config.delim_history
    if not os.getenv("STOP_BY"):
        cg_to_return.stop_by = chatbot_config.stop_by
    if not os.getenv("PREFIX_CONTEXT"):
        cg_to_return.prefix_context = chatbot_config.prefix_context
    if not os.getenv("PREFIX_PROMPT"):
        cg_to_return.prefix_prompt = chatbot_config.prefix_prompt
    if not os.getenv("SUFFIX_PROMPT"):
        cg_to_return.suffix_prompt = chatbot_config.suffix_prompt
    if not os.getenv("TEMPERATURE"):
        cg_to_return.temperature = chatbot_config.temperature
    if not os.getenv("ADD_CONTEXT"):
        cg_to_return.add_context = chatbot_config.add_context
    if not os.getenv("EMBEDDING_MODEL"):
        cg_to_return.embedding_model = chatbot_config.embedding_model
    if not os.getenv("MAX_HISTORY"):
        cg_to_return.max_history = chatbot_config.max_history
    if not os.getenv("THRESHOLD_CONTEXT"):
        cg_to_return.threshold_context = chatbot_config.threshold_context
    if not os.getenv("NUM_RETRIEVE_CONTEXT"):
        cg_to_return.num_retrieve_context = chatbot_config.num_retrieve_context

    return cg_to_return
