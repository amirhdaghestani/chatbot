"""This module handles Elsaticsearch operations"""
from typing import List

import pandas as pd
from elasticsearch import Elasticsearch

from config.elastic_config import ElasticConfig
from logger.ve_logger import VeLogger


class ElasticService:
    """Elasticsearch class"""

    # Initialize logger
    logger = VeLogger()

    def __init__(self, elastic_config: ElasticConfig=None) -> None:
        """Initializer of FaissService class.

        Args:
            elastic_config (ElasticConfig): Necessary configs to be used by class.

        Returns:
            None

        Raises:
            ValueError: When elastic_config is not provided.

        """
        # Check arguments
        if elastic_config is None:
            self.logger.error("Elastic config is None.")
            raise ValueError("Provide elastic_config when initializing class.")

        self.elastic_config = elastic_config
        self.es = self._connect()
        self.database = self._initialize(self.es)

    def _connect(self) -> Elasticsearch:
        """Connect to Elsatic
        
        Args:
            None

        Returns:
            None

        """
        es = Elasticsearch([{'host': self.elastic_config.host, 
                             'port': self.elastic_config.port, 
                             "scheme": self.elastic_config.scheme}],
                             basic_auth=(self.elastic_config.username,
                                         self.elastic_config.password))
        return es

    def _initialize(self, es: Elasticsearch) -> None:
        """Initialize Elastic by adding data into elasticsearch
        
        Args:
            es (Elasticsearch): Elasticsearch database.

        Returns:
            None

        """
        database = pd.read_csv(self.elastic_config.database_file_path,
                               quoting=self.elastic_config.quoting,
                               sep=self.elastic_config.delimiter,
                               escapechar=self.elastic_config.escapechar,
                               keep_default_na=self.elastic_config.keep_default_na)

        # for i in range(len(database)):
        #     row = database.iloc[i]
        #     doc = {
        #         self.elastic_config.query_key: str(row[self.elastic_config.query_key]),
        #         self.elastic_config.answer_key: str(row[self.elastic_config.answer_key]),
        #         self.elastic_config.query_id_key: str(row[self.elastic_config.query_id_key])
        #     }
        #     # index the document
        #     es.index(index=self.elastic_config.index_name, body=doc)

        database = database.set_index(self.elastic_config.query_id_key)
        return database

    def search(self, text: str, num_retrieve: int=10) -> List:
        """Search in Elasticsearch
        
        Args:
            text (str): Input string to search for.
            
        Returns:
            List: Retrieved documents.
            
        """
        # define the search query
        query = {         
            'match': {            
                self.elastic_config.query_key: {
                    "query": text,
                    "fuzziness": self.elastic_config.fuzziness
                }
            }
        }

        # search for documents
        result = self.es.search(index=self.elastic_config.index_name, query=query,
                                size=num_retrieve)

        query_list = []
        score_list = []
        query_id_list = []
        for res in result.body['hits']['hits']:
            query_list.append(res['_source'][self.elastic_config.query_key])
            score_list.append(res['_score'])
            query_id_list.append(int(res['_source'][self.elastic_config.query_id_key]))

        return query_list, score_list, query_id_list
