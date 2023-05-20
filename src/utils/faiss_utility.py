"""This module contains utility function used by faiss service."""
import hashlib


def id_generator(input_str: str=None, limiter: int=7) -> int:
    """Generates unique id based on string input.

    Args:
        input_str (str): Input string to be encoded into unique id.
        limiter (int): Use only this number of input_str.

    Returns:
        int: Generated unique id.

    """
    hash_byte = hashlib.md5(input_str.encode('utf8')).digest()[:limiter]
    return int.from_bytes(hash_byte, byteorder='big')
