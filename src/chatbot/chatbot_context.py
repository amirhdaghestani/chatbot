"""This module extracts context based on the input question"""
import os
from typing import List

from config.faiss_config import FaissConfig
from config.elastic_config import ElasticConfig
from config.vector_config import VectorConfig
from config.chatbot_config import EmbeddingModel
from config.chatbot_config import ChatBotConfig
from faiss_services.faiss_service import FaissService
from elastic_services.elastic_service import ElasticService
from vector_services.vector_service import VectorService
from normalizer.normalizer import Normalizer


class ChatBotContext:
    """This class extracts context based on the input question"""

    def __init__(self, chatbot_config: ChatBotConfig=None) -> None:
        """Initilizer of the class
        
        Args:
            chatbot_config (ChatBotConfig): Necessary configs.

        Returns:
            None

        """
        self.normalizer = Normalizer()
        self.faiss_config, self.vector_config, self.elastic_config = (
            self._get_config(chatbot_config)
        )
        self.num_retrieve_context = chatbot_config.num_retrieve_context

        self.faiss_service = FaissService(self.faiss_config)
        self.vector_service = VectorService(self.vector_config)
        self.elastic_service = ElasticService(self.elastic_config)

    def _get_config(self, chatbot_config: ChatBotConfig):
        faiss_config = FaissConfig()
        elastic_config = ElasticConfig()
        vector_config = VectorConfig()

        if chatbot_config.embedding_model == EmbeddingModel.ADA:
            faiss_config.faiss_file_path = "resources/faiss_ada.idx"
            faiss_config.database_file_path = "resources/database_ada.csv"
            elastic_config.database_file_path = "resources/database_ada.csv"
            faiss_config.vector_size = 1536

            vector_config.api_key = chatbot_config.api_key
            vector_config.model = chatbot_config.embedding_model
        
        elif chatbot_config.embedding_model == EmbeddingModel.ZIBERT:
            faiss_config.faiss_file_path = "resources/faiss_zibert.idx"
            faiss_config.database_file_path = "resources/database_zibert.csv"
            elastic_config.database_file_path = "resources/database_zibert.csv"
            faiss_config.vector_size = 256

            vector_config.model = chatbot_config.embedding_model
            vector_config.model_path = "resources/zibert_v2"
        
        return faiss_config, vector_config, elastic_config
    
    def _vectorize(self, text: str) -> list:
        """Get the embedding of the input.
        
        Args:
            text (str): text to vectorize.
        
        Returns:
            list: Embedding of the input.

        """
        return self.vector_service.get_embedding(text=text)

    def _serach_for_context(self, similar_results_dict: list) -> list:
        """Search for context based on the input vector.
        
        Args:
            similar_results_dict (dict): Dictionary of similar results.
        
        Returns:
            list: list of dictionaries of similar results.

        """
        for i, similar_results in enumerate(similar_results_dict):
            query_id = similar_results['query_id']
            answer = self.faiss_service.database.loc[query_id]['answer']
            similar_results_dict[i]['answer'] = answer
        
        return similar_results_dict

    def _search_similar_questions_vector(self, text: str, num_retrieve: int,
                                         threshold: float) -> list:
        """Search for context based on the input vector.
        
        Args:
            embedding (list): Embedding of the input.
            num_retrieve (int): Number of similar results to be retrieved.
            threshold (float): Threshold to filter similar results.
        
        Returns:
            list: list of dictionaries of similar results.

        """
        embedding = self._vectorize(text)
        similar_results, scores, query_ids = self.faiss_service.search(
            embedding, num_retrieve=num_retrieve)
        
        similar_results = similar_results[0]
        scores = scores[0]
        query_ids = query_ids[0]

        similar_results_dict = [
            {"question": e, "score":scores[i], "query_id":query_ids[i], "tag":"vector"} \
            for i, e in enumerate(similar_results) if scores[i] >= threshold
        ]

        return similar_results_dict

    def _search_similar_questions_elastic(self, text: str, num_retrieve: int,
                                         threshold: float) -> list:
        """Search for similar questions in elasticsearch.
        
        Args:
            embedding (list): Embedding of the input.
            num_retrieve (int): Number of similar results to be retrieved.
            threshold (float): The threshold to filter the similar results.
        
        Returns:
            list: list of dictionaries of similar results.

        """
        similar_results, scores, query_ids = self.elastic_service.search(
            text=text, num_retrieve=num_retrieve)

        similar_results_dict = [
            {"question": e, "score":scores[i], "query_id":query_ids[i], "tag":"elstic"} \
            for i, e in enumerate(similar_results) if scores[i] >= threshold
        ]

        return similar_results_dict
    
    def _mix_similar_questions(self, vector_list: List,
                               elastic_list: List) -> List:
        """Mix two lists.
        
        Args:
            elastic_list (List): List of elastic similar questions.
            vector_list (List): List of vector similar questions.

        Returns:
            List: List of mixed similar questions.

        """
        similar_questions_dict = []
        similar_questions = []

        index_vector = 0
        index_elastic = 0

        while index_vector < len(vector_list) or index_elastic < len(elastic_list):
            if index_vector < len(vector_list) and \
                vector_list[index_vector]['question'] not in similar_questions:
                similar_questions_dict.append(vector_list[index_vector])
                similar_questions.append(vector_list[index_vector]['question'])
            index_vector += 1

            if index_elastic < len(elastic_list) and \
                elastic_list[index_elastic]['question'] not in similar_questions:
                similar_questions_dict.append(elastic_list[index_elastic])
                similar_questions.append(elastic_list[index_elastic]['question'])
            index_elastic += 1
        
        return similar_questions_dict

    def _get_similar_questions(self, text: str, num_retrieve: dict,
                               threshold_vector: float,
                               threshold_elastic: float) -> List:
        """Get the similar questions.
        
        Args:
            text (str): Input text to search for.
            num_retrieve (int): Number of retrieved results.
            threshold (float): The threshold to filter the similar results.
            
        Returns:
            List: List of similar questions.
        
        """
        similar_questions_vector_dict = {}
        similar_questions_elastic_dict = {}
        if 'vector' in num_retrieve.keys() and \
           num_retrieve['vector'] > 0:
            similar_questions_vector_dict = self._search_similar_questions_vector(
                text=text,
                num_retrieve=num_retrieve['vector'], 
                threshold=threshold_vector)
        
        if 'elastic' in num_retrieve.keys() and \
           num_retrieve['elastic'] > 0:
            similar_questions_elastic_dict = self._search_similar_questions_elastic(
                text=text,
                num_retrieve=num_retrieve['elastic'], 
                threshold=threshold_elastic)

        similar_questions_dict = (
            self._mix_similar_questions(vector_list=similar_questions_vector_dict,
                                        elastic_list=similar_questions_elastic_dict)
        )

        return similar_questions_dict

    def get_context(self, text: str, num_retrieve: int=None,
                    threshold_vector: float=0.5, 
                    threshold_elastic: float=0) -> str:
        """Get the similar questions and answers.
        
        Args:
            text (str): Input text to search for.
            num_retrieve (int): Number of retrieved results.
            threshold (float): The threshold to filter the similar results.
            
        Returns:
            str: Generated context.
        
        """
        if not num_retrieve:
            num_retrieve = self.num_retrieve_context
    
        text = self.normalizer.process(text)

        similar_questions_dict = self._get_similar_questions(
            text, 
            num_retrieve=num_retrieve, threshold_vector=threshold_vector,
            threshold_elastic=threshold_elastic)

        similar_questions_dict = self._serach_for_context(
            similar_questions_dict)

        context_str = ""
        for i, similar_questions in enumerate(similar_questions_dict):
            context_str += similar_questions['question'] + "\n"
            context_str += similar_questions['answer']
            if i != len(similar_questions_dict) - 1:
                context_str += "\n\n"

        return context_str
