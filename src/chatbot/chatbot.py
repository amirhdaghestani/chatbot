"""This module contains ChatBot class which is used for processing user input 
and handling openai API calls"""
import os

import openai

from logger.ve_logger import VeLogger
from config.chatbot_config import ChatBotConfig

__CHATMODELS__ = ["gpt-3.5-turbo"]


class ChatBot:
    """Chat bot class """

    # Initialize logger
    logger = VeLogger()

    def __init__(self, chatbot_config: ChatBotConfig=None) -> None:
        """Initializer of ChatBot class.
        
        Args:
            chatbot_config (ChatBotConfig): Necessary configs to be used by class.

        Returns:
            None
        
        Raises:
            ValueError: when chatbot_config is not provided.

        """
        # Check arguments
        if chatbot_config is None:
            self.logger.error("chatbot config is None.")
            raise ValueError("Provide chatbot_config when initializing class.")

        if chatbot_config.api_key is None:
            self.logger.error("API key is None.")
            raise ValueError(
                "Provide API key when initializing class. You can set the " \
                "enviroment variable `OpenAI_API_KEY` to your API key.")

        # Set OpenAI API key and variables
        openai.api_key = chatbot_config.api_key
        self.chat_engine = chatbot_config.chat_engine
        self.max_tokens = chatbot_config.max_tokens
        self.num_responses = chatbot_config.num_responses
        self.stop_by = chatbot_config.stop_by
        self.temperature = chatbot_config.temperature
        self.bot_description = chatbot_config.bot_description

        self.messages = self._init_messages()

    def _init_messages(self):
        """Initialize messages
        
        Returns:
            list: messages list.

        """
        if self.chat_engine in __CHATMODELS__:
            messages = [
                {"role": "system", "content": self.bot_description}
            ]
        else:
            messages = self.bot_description

        return messages

    def _chat_completion(self, message: str=None):
        """Function to call chat completion from openai.
        
        Args:
            messages (str): messages for API call.
        
        Returns:
            list: API response.

        """
        # Call API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=message
        )

        return [response.choices[0].message.content]

    def _prompt_completion(self, prompt: str=None):
        """Function to call prompt completion from openai.
        
        Args:
            prompt (str): prompts for API call.
        
        Returns:
            list: API response.

        """
        # Call API
        response = openai.Completion.create(
            engine=self.chat_engine,
            prompt=prompt,
            max_tokens=self.max_tokens,
            n=self.num_responses,
            stop=self.stop_by,
            temperature=self.temperature,
        )

        response_text = []
        for choice in response.choices:
            response_text.append(choice.text)

        return response_text

    def _create_prompt(self, message: str=None):
        """Function to construct prompt.

        Args:
            message (str): Message to get response for.

        Returns:
            str: Generated response.

        """
        if self.chat_engine in __CHATMODELS__:
            created_message = [
                {"role": "system", "content": self.bot_description},
                {"role": "user", "content": message}
            ]
        else:
            created_message = self.bot_description + "\nQuestion:\n" \
                            + message + "\nAnswer:"
        return created_message

    def generate_response(self, message: str=None):
        """Function to generate response from model.
        
        Args:
            message (str): Message to get response for.
        
        Returns:
            str: Generated response.
        
        Raises:
            ValueError: when message is not provided.

        """
        # Check arguments
        if message is None:
            self.logger.error("message is None.")
            raise ValueError("Provide message when calling this function.")

        prompt = self._create_prompt(message=message)

        if self.chat_engine in __CHATMODELS__:
            response = self._chat_completion(message=prompt)
        else:
            response = self._prompt_completion(prompt=prompt)

        return response
