"""This module contains ChatBot class which is used for processing user input 
and handling openai API calls"""
import os
import copy

import openai

from logger.ve_logger import VeLogger
from config.chatbot_config import ChatBotConfig
from chatbot.chatbot_context import ChatBotContext
from chatbot.chatbot_post_process import ChatBotPostProcess


__CHATMODELS__ = [
    "gpt-3.5-turbo", 
    "gpt-4"
]


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
        self.delim_botdesc = chatbot_config.delim_botdesc
        self.delim_context = chatbot_config.delim_context
        self.delim_history = chatbot_config.delim_history
        self.prefix_context = chatbot_config.prefix_context
        self.prefix_prompt = chatbot_config.prefix_prompt
        self.suffix_prompt = chatbot_config.suffix_prompt
        self.add_context = chatbot_config.add_context
        self.max_history = chatbot_config.max_history
        self.threshold_context_vector = chatbot_config.threshold_context_vector
        self.threshold_context_elastic = chatbot_config.threshold_context_elastic
        self.num_retrieve_context = chatbot_config.num_retrieve_context
        self.post_process = chatbot_config.post_process

        self.messages = self._init_messages()
        if chatbot_config.add_context:
            self.chatbot_context = ChatBotContext(chatbot_config=chatbot_config)

        self.chatbot_post_process = ChatBotPostProcess(
            chatbot_config=chatbot_config)

    def _init_messages(self):
        """Initialize messages
        
        Returns:
            list: messages list.

        """
        messages = [
            {"role": "system", "content": self.bot_description}
        ]
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
            model=self.chat_engine, 
            messages=message,
            max_tokens=self.max_tokens,
            n=self.num_responses,
            stop=self.stop_by,
            temperature=self.temperature,
        )

        response_text = []
        for choice in response.choices:
            response_text.append(choice.message.content)

        return response_text

    def _prompt_completion(self, prompt: str=None):
        """Function to call prompt completion from openai.
        
        Args:
            prompt (str): prompts for API call.
        
        Returns:
            list: API response.

        """
        # Call API
        response = openai.Completion.create(
            model=self.chat_engine,
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

    def _create_prompt(self, message: str=None, context: str=None):
        """Function to construct prompt.

        Args:
            message (str): Message to get response for.
            context (str): Context to answer from.

        Returns:
            str: Generated response.

        """
        if context:
            context = self.prefix_context + context
        else:
            context = ""

        messages = copy.copy(self.messages)

        messages[0] = {
            "role": "system", "content": (self.bot_description \
                                          + self.delim_botdesc + context)
        }
        
        messages.append({
            "role": "user", "content": message
        })

        return messages
    
    def _convert_to_prompt_models(self, messages: str=None):
        """Convert the message to prompt based models
        
        Args:
            messages (str): message to be converted.
        
        Returns:
            str: Converted message.

        """
        prompt_str = ""
        for i, message in enumerate(messages):
            if message['role'] == "system":
                prompt_str += message['content'] + self.delim_context
            elif message['role'] == "user":
                prompt_str += self.prefix_prompt + message['content']
            elif message['role'] == "assistant":
                prompt_str += self.suffix_prompt + message['content'] \
                    + self.delim_history
        prompt_str += self.suffix_prompt

        return prompt_str
    
    def _choose_best_response(self, response: list):
        """To choose the best response.
        
        Args:
            response (list): list of all responses of the model.
        
        Returns:
            str: The best response.

        """
        # Not properly implemented yet.
        return response[0]

    def _post_process(self, response: list, context: str=None, 
                      message: str=None):
        """To apply post process.

        Args:
            response (list): list of all responses of the model.

        Returns:
            list: Apllied post process responses.

        """
        if len(self.post_process) == 0:
            return response

        responses = []
        for post_process_item in self.post_process:
            responses.append(
                self.chatbot_post_process.post_process[post_process_item](
                    prompt=message, context=context, answers=response
                )
            )
        return responses

    def generate_response(self, message: str=None, return_prompt: bool=False,
                          return_context: bool=False):
        """Function to generate response from model.

        Args:
            message (str): Message to get response for.
            add_context (bool): Whether to add context or not.
            return_prompt (bool): Whether to return prompt or not.
            return_context (bool): Whether to return context or not.

        Returns:
            list: Generated response.
            list: Prompt of request.
            list: Retrieved context.

        Raises:
            ValueError: when message is not provided.

        """
        # Check arguments
        if message is None:
            self.logger.error("message is None.")
            raise ValueError("Provide message when calling this function.")

        if len(self.messages) > 2 * self.max_history:
            del self.messages[1]
            del self.messages[1]

        context = None
        if self.add_context:
            context = self.chatbot_context.get_context(
                message,
                num_retrieve=self.num_retrieve_context,
                threshold_vector=self.threshold_context_vector,
                threshold_elastic=self.threshold_context_elastic)

        messages = self._create_prompt(message=message, context=context)

        if self.chat_engine in __CHATMODELS__:
            response = self._chat_completion(message=messages)
        else:
            messages_converted = self._convert_to_prompt_models(messages)
            response = self._prompt_completion(prompt=messages_converted)

        post_process = self._post_process(response=response, context=context, 
                                          message=message)
        best_response = self._choose_best_response(post_process)
        
        self.messages = messages
        self.messages.append(
            {"role": "assistant", "content": best_response}
        )

        return_tuple = best_response

        if return_prompt or return_context:
            return_tuple = (return_tuple,)

        if return_prompt:
            return_tuple += (messages,)

        if return_context:
            return_tuple += (context,)

        return return_tuple
