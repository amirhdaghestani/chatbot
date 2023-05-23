"""This module contains necessary configs for vector service"""
import os
from enum import Enum


class EmbeddingModel(Enum):
    ADA = "text-embedding-ada-002"
    ZIBERT = "zibert_v2"


class VectorConfig:
    """Necessary configs for vector service.

    Attributes:
        api_key [required] (str): OpenAI api key.

    """
    api_key = str(os.getenv("OPENAI_API_KEY")) \
              if os.getenv("OPENAI_API_KEY") else None
    model = EmbeddingModel(os.getenv("VECTOR_MODEL")) \
            if os.getenv("VECTOR_MODEL") else EmbeddingModel.ZIBERT
    model_path = str(os.getenv("MODEL_PATH")) \
                 if os.getenv("MODEL_PATH") else "resources/zibert_v2"
    
    def __init__(self, model: EmbeddingModel=None, api_key: str=None,
                 model_path: str=None) -> None:
        if model:
            self.model = model
        if model_path:
            self.model_path = model_path
        if api_key:
            self.api_key = api_key