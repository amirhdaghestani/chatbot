"""This module handles normalizing and expansion"""
import re

import hazm
from num2words import num2words

from logger.ve_logger import VeLogger

class Normalizer:
    """Normalizer and Expansion class"""

    # Initialize logger
    logger = VeLogger()

    def __init__(self) -> None:
        """Initializer of Normalizer"""
        self.norm = hazm.Normalizer()
        self.num_word = self._generate_num_word()

    def _num2word(self, number) -> str:
        """Convert number to word.
        
        Args:
            number (int, float): Input number.
        
        Returns:
            str: Converted number.

        """
        return num2words(number, lang ='fa')

    def _generate_num_word(self, max_num: int=1000) -> tuple:
        """Generate numbers in number and word format
        
        Args:
            max_num (int): maximum number to generate.
        
        Returns:
            tuple: numbers in number and word format.

        """
        word = []

        for i in range(max_num):
            word.append(self._num2word(i))
        
        return word

    def expansion(self, text: str) -> str:
        numbers_in_text = set([e for e in re.findall(r'[\d\.\d]+', text) if e != "."])
        
        for number in numbers_in_text:
            text = text.replace(f" {number} ", f" {number} ({self._num2word(float(number))}) ")
        
        for i, word in enumerate(self.num_word):
            text = text.replace(f" {word} ", f" {word} ({i}) ")
        
        return text

    def normalize(self, text: str) -> str:
        """Normalize input text.
        
        Args:
            text (str): Input text.

        Returns:
            str: Normalized input text
        """
        return self.norm.normalize(text)
    
    def process(self, text: str) -> str:
        """Expand and normalize query.
        
        Args:
            text (str): Input text.

        Returns:
            str: Expanded and normalized query.

        """
        text = self.expansion(text)
        text = self.normalize(text)

        return text
