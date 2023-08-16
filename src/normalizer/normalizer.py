"""This module handles normalizing and expansion"""
import re

import hazm
from num2words import num2words

from logger.ve_logger import VeLogger


class Normalizer:
    """Normalizer and Expansion class"""

    # Initialize logger
    logger = VeLogger()

    WORD_EXPANSION = {
        "گیگ": "گیگابایت",
        "گیگی": "گیگابایتی",
        "پیش آواز": "آوای انتظار",
        "پیشواز": "آوای انتظار",
        "پیش اواز": "آوای انتظار",
        "پیشاواز": "آوای انتظار",
        "پیشآواز": "آوای انتظار",
    }

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

    def _num_expansion(self, text: str) -> str:
        """Expand numbers into words and vice versa.

        Args:
            text (str): Input string.

        Returns:
            str: Expanded text.

        """
        numbers_in_text = set([e for e in re.findall(r'[\d\.\d]+', text) \
                               if e != "." and e.count(".") <= 1])

        for number in numbers_in_text:
            text = text.replace(f" {number} ", f" {number} ({self._num2word(float(number))}) ")

        for i, word in enumerate(self.num_word):
            text = text.replace(f" {word} ", f" {word} ({i}) ")

        return text

    def _word_expansion(self, text: str) -> str:
        """Expand words.

        Args:
            text (str): Input string.

        Returns:
            str: Expanded text.

        """

        for word, replace in self.WORD_EXPANSION.items():
            text = text.replace(f" {word} ", f" {replace} ")

        return text

    def expansion(self, text: str) -> str:
        """Expand the input text.

        Args:
            text (str): Input string.

        Returns:
            str: Expanded text.

        """
        text = self._num_expansion(text=text)
        text = self._word_expansion(text=text)

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
        text = " " + text + " "
        text = self.expansion(text)
        text = self.normalize(text)

        return text
