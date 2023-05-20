"""This module searches in Faiss"""
# TODO: Add detailed log.
# pylint: disable=E0401, E0611
from typing import List

import faiss
import pandas as pd
import numpy as np

from logger.ve_logger import VeLogger
from config.faiss_config import FaissConfig

class FaissSearch:
    """This class implements search in Faiss."""

    # Initialize logger
    logger = VeLogger()

    def __init__(self, faiss_config: FaissConfig=None, index: faiss=None,
                 database: pd=None) -> None:
        """Initializer method

        Args:
            index (faiss): Input faiss.
            database (pd): Input database.

        Returns:
            None

        Raises:
            ValueError: When 'index' is not provided.

        """
        # Check arguments
        if faiss_config is None:
            self.logger.error("Faiss config is None.")
            raise ValueError("Provide faiss_config when initializing class.")

        if index is None:
            self.logger.error("The input 'index' must be of type 'faiss'")
            raise ValueError("The input 'index' must be of type 'faiss'")

        if database is None:
            self.logger.error("The input 'database' must be of type 'pd'")
            raise ValueError("The input 'database' must be of type 'pd'")

        # Initialize variable
        self.faiss_config = faiss_config
        self.index = index
        self.database = database

    def _search_index(self, query_vector: List, num_retrieve: int=10) -> List:
        """Search Faiss.

        Args:
            query_vector (list): Input vector to search.
            num_retrieve (int): Number of results to retrive [Default: 10].

        Returns:
            list: Retrived scores and ids.

        Raises:
            Exception: When searching in faiss encounters a problem.

        """
        try:
            scores, query_ids = self.index.search(
                np.array(query_vector, dtype=np.single),
                num_retrieve,
            )
            query_ids = query_ids.tolist()
            scores = scores.tolist()
        except Exception as exception:
            self.logger.error("Error searching in Faiss.",
                              extra={"exc_message": exception},
                              exc_info=exception)
            raise Exception("Error searching in Faiss.", exception) \
            from exception

        for index, query_id in enumerate(query_ids):
            valid_rows = [
                idx for idx, row_id in enumerate(query_id) \
                if row_id != -1
            ]
            query_ids[index] = [query_ids[index][idx] for idx in valid_rows]
            scores[index] = [scores[index][idx] for idx in valid_rows]

        return query_ids, scores

    def _search_database(self, query_ids: List) -> List:
        """Search databse.

        Args:
            query_ids (List): query_ids to retrieve from database.

        Returns:
            list: Retrieved queries.

        Raises:
            Exception: When searching in database encounters problem.

        """
        similar_queries = []
        for query_id in query_ids:
            try:
                temp_result = self.database.loc[query_id]
                temp_result = temp_result[self.faiss_config.query_key].tolist()
                similar_queries.append(temp_result)
            except Exception as exception:
                self.logger.error("Error searching in database.",
                                  extra={"exc_message": exception},
                                  exc_info=exception)
                raise Exception("Error searching in database.", exception) \
                from exception

        return similar_queries

    def search(self, query_vector: List, num_retrieve: int=10) -> List:
        """Search query.

        Args:
            query_vector (list): Input vector to search.
            num_retrieve (int): Number of results to retrive [Default: 10].

        Returns:
            list: Retrived results.

        Raises:
            ValueError: When input 'query_vec' is not of type 'List'.

        """
        # Check input type
        if not isinstance(query_vector, List):
            self.logger.error("Input query must be of type 'List'.")
            raise ValueError("Input query must be of type 'List'.")

        if not isinstance(query_vector[0], List):
            query_vector = [query_vector]

        # Return results from Faiss
        query_ids, scores = self._search_index(
            query_vector=query_vector,
            num_retrieve=num_retrieve
        )

        similar_queries = self._search_database(query_ids=query_ids)

        return similar_queries, scores, query_ids
