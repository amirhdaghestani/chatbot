"""This module implemetns post process box"""
import random

import openai
import tiktoken

from config.chatbot_config import ChatBotConfig
from faiss_services.faiss_service import FaissService
from vector_services.vector_service import VectorService


class ChatBotPostProcess:
    """This class extracts context based on the input question"""

    def __init__(self, chatbot_config: ChatBotConfig=None) -> None:
        """Initilizer of the class
        
        Args:
            path (str): Path of the database files.

        Returns:
            None.

        """
        self.classification_prompt_template = (
            "کانتکست: {}\n\n###\n\nسوال: {}\n\n###\n\nپاسخ: {}\n\n###\n\nصحت: "
        )
        self.classification_dontknow = [
            "متاسفانه سوال شما در حوزه‌ی اطلاعات من نیست، لطفا با شماره‌ی 9990 تماس بگیرید و یا از طریق لینک با پشتیبانی انسانی تماس فرمایید.",
            "با عرض پوزش پاسخ سوال شما در حیطه‌ی اطلاعات من نیست. لطفا با شماره‌های پشتیبانی همراه اول به شماره‌ی 9990 تماس بگیرید.",
            "ببخشید، پاسخ سوال شما را نمی‌دانم. از شماره‌ی پشتیبانی 9990 همراه اول کمک بگیرید."
        ]
        self.post_process = {
            "classification": self.classification_process
        }
        self.post_process_params = chatbot_config.post_process_params

        self.enc = tiktoken.encoding_for_model("ada")

    def _classification_inference(self, prompt: str):
        """Inference classification model.
        
        Args:
            prompt (str): Input prompt.

        Returns:
            str: Result of classification.

        """
        response = openai.Completion.create(
            model="ada:ft-personal-2023-06-19-08-49-19",
            prompt=prompt,
            max_tokens=1,
            temperature=0,
            logprobs=2,
            logit_bias={3763: 100, 645: 100}
        )

        log_probs = []
        for choice in response.choices:
            log_probs.append(
                choice.logprobs['top_logprobs'][0]
            )

        return log_probs

    def _prompt_creation(self, prompt_template: str, prompt: str, 
                         context: str, answer: str, max_tokens: int=2048):
        """Creat prompt and reduce number of tokens, if necessary.
        
        Args:
            prompt (str): Question asked.
            context (str): Context provided.
            answers (list): List of generated answers by the model.
        
        Returns:
            str: reduced (if needed) formatted string

        """
        prompt_str = prompt_template.format(context, prompt, answer)
        tokenized_prompt = self.enc.encode(prompt_str)
        if len(tokenized_prompt) < max_tokens:
            return prompt_str
        else:
            len_to_reduce = len(tokenized_prompt) - 2047
            context = self.enc.decode(self.enc.encode(context)[len_to_reduce:])
            return prompt_template.format(context, prompt, answer)

    def classification_process(self, prompt: str, context: str, answers: list, 
                               threshold: float=None):
        """Post process classification for truthfulness.
        
        Args:
            prompt (str): Question asked.
            context (str): Context provided.
            answers (list): List of generated answers by the model.

        Returns:
            str: Either the best answer or don't know answer.

        """
        # Set arguments
        if threshold is None:
            threshold = self.post_process_params['classification_threshold']

        log_probs = []
        for answer in answers:
            log_prob = self._classification_inference(
                self._prompt_creation(
                    prompt_template=self.classification_prompt_template,
                    context=context, prompt=prompt, answer=answer
                )
            )
            log_probs.append(log_prob[0])
        max_log_probs_yes_set = [log_prob[' yes'] for log_prob in log_probs]
        max_log_probs_yes = max(max_log_probs_yes_set)
        index_max_log_probs_yes = max_log_probs_yes_set.index(max_log_probs_yes)

        if max_log_probs_yes < threshold:
            return random.sample(self.classification_dontknow, 1)[0]
        
        return answers[index_max_log_probs_yes]
