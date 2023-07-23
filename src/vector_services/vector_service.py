"""This module implements vector service to extract embeddings from raw text"""
import time

import numpy as np
import openai
import hazm
from transformers import AutoTokenizer, AutoModel

from config.vector_config import VectorConfig
from config.vector_config import EmbeddingModel


class VectorService:
    """The class to vectorize input text"""

    OPENAI_MODELS = [
        EmbeddingModel.ADA
    ]

    TRANSFORMER_MODELS = [
        EmbeddingModel.ZIBERT
    ]

    def __init__(self, vector_service_config: VectorConfig=None) -> None:
        """Initializer function of class

        Args:
            model (EmbeddingType): model type to extract embeddings.

        Returns:
            None

        """
        # Check arguments
        if vector_service_config is None:
            self.logger.error("vector service config is None.")
            raise ValueError(
                "Provide vector_service_config when initializing class.")
        
        self.model = vector_service_config.model
        self.normalizer = hazm.Normalizer()

        if self.model in self.OPENAI_MODELS:
            openai.api_key = vector_service_config.api_key
        
        elif self.model in self.TRANSFORMER_MODELS:
            self.tokenizer = AutoTokenizer.from_pretrained(
                vector_service_config.model_path)
            self.model_tr = AutoModel.from_pretrained(
                vector_service_config.model_path)

    def _request_openai(self, text: str, model: EmbeddingModel, 
                        retry_times: int=5, sleep_time: int=5):
        """Retry hnadler for openai request"""
        for retry in range(retry_times):
            try:
                result = openai.Embedding.create(
                    input = [text], model=model.value)['data'][0]['embedding']
            except:
                time.sleep(sleep_time)
                continue
            break
        return result
    def _openai_embeddings(self, text: str, model: EmbeddingModel):
        """Generate embedding from OpenAI
        
        Args:
            text (str): Input text.
            model (str): Model type to extract feature from.

        Returns:
            list: Embedding of the input.
            
        """
        text = text.replace("\n", " ")
        return self._request_openai(text, model=model, retry_times=5, 
                                    sleep_time=5)

    def _transformers_embeddings(self, text: str):
        """Generate embedding from transformers models
        
        Args:
            text (str): Input text.

        Returns:
            list: Embedding of the input.
        """
        tokens = self.tokenizer(text, return_tensors='pt')
        vector = self.model_tr(**tokens)
        vector = vector['last_hidden_state'][0, 0, :].detach().numpy()
        vector_norm = np.divide(vector, np.linalg.norm(vector))
        return list(vector_norm)


    def get_embedding(self, text: str):
        """Generate embedding
        
        Args:
            text (str): input text.

        Returns:
            List: embedding of the input.

        """
        text = self.normalize(text)
        if self.model in self.OPENAI_MODELS:
            return self._openai_embeddings(text=text, model=self.model)
        elif self.model in self.TRANSFORMER_MODELS:
            return self._transformers_embeddings(text=text)
