"""This module extracts context based on the input question"""
import os

from config.faiss_config import FaissConfig
from config.vector_config import VectorConfig
from faiss_services.faiss_service import FaissService
from vector_services.vector_service import VectorService


class ChatBotContext:
    """This class extracts context based on the input question"""

    def __init__(self, path: str=None) -> None:
        """Initilizer of the class
        
        Args:
            path (str): Path of the database files.

        Returns:
            None.

        """
        self.faiss_config = FaissConfig()
        self.vector_config = VectorConfig()

        if path:
            self.faiss_config.faiss_file_path = os.path.join(path, "faiss.idx")
            self.faiss_config.database_file_path = os.path.join(
                path, "database.csv")

        self.faiss_service = FaissService(self.faiss_config)
        self.vector_service = VectorService(self.vector_config)

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

    def _search_similar_questions(self,  embedding: list, num_retrieve: int,
                                  threshold: float) -> list:
        """Search for context based on the input vector.
        
        Args:
            embedding (list): Embedding of the input.
            num_retrieve (int): Number of similar results to be retrieved.
            threshold (float): Threshold to filter similar results.
        
        Returns:
            list: list of dictionaries of similar results.

        """
        similar_results, scores, query_ids = self.faiss_service.search(
            embedding, num_retrieve=num_retrieve)
        
        similar_results = similar_results[0]
        scores = scores[0]
        query_ids = query_ids[0]

        similar_results_dict = [
            {"question": e, "score":scores[i], "query_id":query_ids[i]} \
            for i, e in enumerate(similar_results) if scores[i] >= threshold
        ]

        return similar_results_dict

    def get_context(self, text: str, num_retrieve: int=10,
                    threshold: float=0.5) -> str:
        """Get the similar questions and answers.
        
        Args:
            text (str): Input text to search for.
            num_retrieve (int): Number of retrieved results.
            threshold (float): The threshold to filter the similar results.
            
        Returns:
            str: Generated context.
        
        """

        embedding = self._vectorize(text)
        similar_questions_dict = self._search_similar_questions(
            embedding, 
            num_retrieve=num_retrieve, 
            threshold=threshold)
        similar_questions_dict = self._serach_for_context(
            similar_questions_dict)

        context_str = ""
        for similar_questions in similar_questions_dict:
            context_str += similar_questions['question'] + "\n"
            context_str += similar_questions['answer'] + "\n\n"
        
        return context_str
