"""This module initializes Faiss"""
# TODO: Add detailed log.
# pylint: disable=E0401, E0611
from typing import List, Union

import faiss
import pandas as pd
import numpy as np

from logger.ve_logger import VeLogger
from config.faiss_config import FaissConfig
from utils.faiss_utility import id_generator

class FaissInitializer:
    """This class implements Faiss initializer."""

    # Initialize logger
    logger = VeLogger()

    def __init__(self, data: Union[List[dict], dict],
                 faiss_config: FaissConfig=None) -> None:
        """Initializer method

        Args:
            faiss_config (FaissConfig): Input configs to be used.

        Returns:
            None

        Raises:
            ValueError: When 'fais_config' is not provided.

        """
        # Check arguments
        if faiss_config is None:
            self.logger.error("Faiss config is None.")
            raise ValueError("Provide faiss_config when initializing class.")

        # Initialize variable
        self.faiss_config = faiss_config

        # Initialize and save index and database
        self._init(data=data)

    def _generate_id_from_query(self, query: str=None) -> int:
        """Generate unique query_id based on query.

        Args:
            query (str): Input query to generate a unique id.

        Returns:
            int: Generated unique id.

        """
        return id_generator(query, self.faiss_config.id_generator_limiter)

    def _add_query_id(self, data: List[dict]) -> None:
        """Add ids to data list.

        Args:
            data (List[dict]): Input data dict (complex object, passed by ref).

        Returns:
            None

        Raises:
            ValueError: When data dict doesn't have query key.

        """
        for index, data_dict in enumerate(data):
            query = data_dict.get(self.faiss_config.query_key)

            # Check if query exists.
            if query is None:
                self.logger.error(f"Data row {index} has no query.")
                raise ValueError(f"Data row {index} has no query.")

            # Generate unique ids from queries.
            data_dict[self.faiss_config.id_key] = (
                self._generate_id_from_query(query)
            )

    def _dict2list(self, data: List[dict]) -> int:
        """Generate data lists.

        Args:
            data (List[dict], dict): List (or a single) dict
                of data which contains query with the query_key and vector
                with the key vector_key.

        Returns:
            List: 'queries', 'vectors', and 'query_ids' lists.

        """
        # Define lists to store data
        query_list = []
        vector_list = []
        query_ids = []
        for index, data_dict in enumerate(data):
            query = data_dict.get(self.faiss_config.query_key)
            vector = data_dict.get(self.faiss_config.vector_key)
            query_id = data_dict.get(self.faiss_config.id_key)

            # Check if vector exists.
            if vector is None:
                self.logger.error(f"Data row {index} has no vector.")
                raise ValueError(f"Data row {index} has no vector.")

            # Append to a list, to add to faiss and database in batch or
            # Change values if query_id is repeated
            if query_id not in query_ids:
                query_ids.append(query_id)
                query_list.append(query)
                vector_list.append(vector)
            else:
                index = query_ids.index(query_id)
                vector_list[index] = vector
                query_list[index] = query

        return query_list, vector_list, query_ids

    def _init(self, data: Union[List[dict], dict]) -> None:
        """Initializes faiss and saves index files.

        Args:
            data (List[dict], dict): List (or a single) dict
                of data which contains query with the query_key and vector
                with the key vector_key.

        Returns:
            None.

        """
        if isinstance(data, dict):
            data = [data]

        # Add ids to data list
        self._add_query_id(data=data)

        # Define lists to store data
        query_list, vector_list, query_ids = self._dict2list(data=data)

        database = pd.DataFrame(query_list,
                                columns=[self.faiss_config.query_key],
                                index=query_ids)
        database.index.name = self.faiss_config.id_key

        # Initialize Faiss (pylint: disable=E1120)
        quantizer = faiss.IndexFlatIP(self.faiss_config.vector_size)
        index = faiss.IndexIVFFlat(quantizer,
                                   self.faiss_config.vector_size,
                                   self.faiss_config.nlist,
                                   faiss.METRIC_INNER_PRODUCT)

        index.train(np.array(vector_list, dtype=np.single))
        index.nprobe = 10
        index.add_with_ids(np.array(vector_list, dtype=np.single),
                           np.array(query_ids,
                           dtype='int64'))
        index.set_direct_map_type(faiss.DirectMap.Hashtable)

        faiss.write_index(index, self.faiss_config.faiss_file_path)
        database.to_csv(self.faiss_config.database_file_path,
                        quoting=self.faiss_config.quoting,
                        sep=self.faiss_config.delimiter,
                        escapechar=self.faiss_config.escapechar)
