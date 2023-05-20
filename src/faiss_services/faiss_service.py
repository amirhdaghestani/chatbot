"""This module contains faiss class which handles faiss operations"""
# TODO: Add detailed log.
# pylint: disable=E0401, E0611
from typing import Union, List
import copy
import time
import threading
import os

import faiss
import pandas as pd
import numpy as np

from logger.ve_logger import VeLogger
from config.faiss_config import FaissConfig
from utils.faiss_utility import id_generator
from utils.rclone import RClone
from .faiss_trainer import FaissTrainer
from .faiss_search import FaissSearch
from .faiss_initializer import FaissInitializer


class FaissService:
    """Faiss class"""

    # Initialize logger
    logger = VeLogger()

    def __init__(self, faiss_config: FaissConfig=None) -> None:
        """Initializer of FaissService class.

        Args:
            faiss_config (FaissConfig): Necessary configs to be used by class.

        Returns:
            None

        Raises:
            ValueError: When faiss_config is not provided.

        """
        # Check arguments
        if faiss_config is None:
            self.logger.error("Faiss config is None.")
            raise ValueError("Provide faiss_config when initializing class.")

        # Init variables
        self.faiss_config = faiss_config
        self.save_flag = False
        self.load_flag = False
        self.faiss_trainer = FaissTrainer(self.faiss_config)

        # Initialize rclone if enabled
        if self.faiss_config.ceph_enable:
            self.rclone = self._init_rclone(self.faiss_config.ceph_config)

        # Load faiss and database
        self.index, self.database = self._load()

        # Load faiss search
        self.faiss_search = FaissSearch(self.faiss_config, self.index,
                                        self.database)

    def _load(self, faiss_file_path: str=None,
              database_file_path: str=None) -> None:
        """Load faiss and database from path.

        Returns:
            None

        Raises:
            RuntimeError: When faiss or database file doesn't exists.

        """
        # Initialize variable if not given by argument.
        if faiss_file_path is None:
            faiss_file_path = self.faiss_config.faiss_file_path

        if database_file_path is None:
            database_file_path = self.faiss_config.database_file_path

        self._download_from_ceph(faiss_file_path=faiss_file_path,
                                 database_file_path=database_file_path)

        # Load faiss and databse
        try:
            index = faiss.read_index(self.faiss_config.faiss_file_path)
            database = pd.read_csv(self.faiss_config.database_file_path,
                                   quoting=self.faiss_config.quoting,
                                   sep=self.faiss_config.delimiter,
                                   escapechar=self.faiss_config.escapechar,
                                   keep_default_na=self.faiss_config.keep_default_na)
            database = database.set_index(self.faiss_config.id_key)
        except RuntimeError as runtime_error:
            self.logger.error("File does not exist. This might happend when" \
                              " local file of faiss or database is not" \
                              " found. Check your configs and make sure it" \
                              "  points to the right location.",
                              extra={"exc_message": runtime_error},
                              exc_info=runtime_error)
            raise RuntimeError(f"{self.faiss_config.database_file_path} or" \
                               f" {self.faiss_config.faiss_file_path} does" \
                                " not exist.") from runtime_error

        return index, database

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

    def _duplicated_ids(self, data: List[dict]) -> List[int]:
        """Returns a list of duplicated ids.

        Args:
            data (List[dict]): Input data dict (complex object, passed by ref).

        Returns:
            List[int]: list of duplicated ids.

        """
        duplicated_ids = []

        for data_dict in data:
            query_id = data_dict.get(self.faiss_config.id_key)

            # Check if query_id is duplicated.
            if (
                query_id in self.database.index and
                query_id not in duplicated_ids
            ):
                duplicated_ids.append(query_id)

        return duplicated_ids

    def _remove_vectors_by_id(self, ids: List[int]) -> None:
        """Remove vectors by ids.

        Args:
            ids (List[int]): ids to be removed from faiss and database.

        Returns:
            None

        Raises:
            KeyError: If id doesn't exist in database and faiss.

        """
        # Remove vectors from faiss and index. (pylint: disable=W0201)
        try:
            self.index.remove_ids(np.array(ids, dtype='int64'))
            self.database = self.database.drop(ids)
        except KeyError as key_error:
            self.logger.error("Failed to remove ids.",
                              extra={"exc_message": key_error},
                              exc_info=key_error)
            raise KeyError("Failed to remove ids.") from key_error

    def _save(self, faiss_file_path: str=None, database_file_path: str=None,
              index: faiss=None, database: pd=None) -> None:
        """Private function to save faiss and database to file

        Args:
            faiss_file_path (str): path to save faiss.
            database_file_path (str): path to save database.
            index (faiss): faiss to be saved.
            database (pd): database to be saved.

        Returns:
            None

        """
        if index is None:
            index = self.index

        if database is None:
            database = self.database

        if faiss_file_path is None:
            faiss_file_path = self.faiss_config.faiss_file_path

        if database_file_path is None:
            database_file_path = self.faiss_config.database_file_path

        faiss.write_index(index, faiss_file_path)
        database.to_csv(database_file_path, quoting=self.faiss_config.quoting,
                        sep=self.faiss_config.delimiter,
                        escapechar=self.faiss_config.escapechar)

        self._upload_to_ceph(faiss_file_path=faiss_file_path,
                             database_file_path=database_file_path)

    def _upload_to_ceph(self, faiss_file_path: str=None,
                        database_file_path: str=None) -> None:
        """Upload to ceph bucket.

        Args:
            faiss_file_path (str): path to save faiss.
            database_file_path (str): path to save database.

        Returns:
            None

        """
        ceph_faiss_file_path = os.path.join(self.faiss_config.ceph_index,
                                            os.path.dirname(faiss_file_path))
        ceph_database_file_path = os.path.join(self.faiss_config.ceph_index,
                                               os.path.dirname(database_file_path))
        if self.faiss_config.ceph_enable and self.rclone:
            self.rclone.copy(faiss_file_path,
                             ceph_faiss_file_path)
            self.rclone.copy(database_file_path,
                             ceph_database_file_path)

    def _download_from_ceph(self, faiss_file_path: str=None,
                            database_file_path: str=None) -> None:
        """Download from ceph bucket.

        Args:
            faiss_file_path (str): path to save faiss.
            database_file_path (str): path to save database.

        Returns:
            None

        """
        ceph_faiss_file_path = os.path.join(self.faiss_config.ceph_index,
                                            faiss_file_path)
        ceph_database_file_path = os.path.join(self.faiss_config.ceph_index,
                                               database_file_path)
        local_faiss_file_path = os.path.dirname(faiss_file_path)
        local_database_file_path = os.path.dirname(database_file_path)

        if self.faiss_config.ceph_enable and self.rclone:
            self.rclone.copy(ceph_faiss_file_path,
                             local_faiss_file_path)
            self.rclone.copy(ceph_database_file_path,
                             local_database_file_path)

    def _init_rclone(self, config_rclone):
        """Initialize rclone.

        Create an instance from rclone class with given config

        Args:
            config_rclone (string): config needed to init rclone.

        Raises:
            ValueError: rclone fail to connect server.

        Returns:
            RClone: rclone object.

        """
        rclone = RClone(config_rclone)
        if rclone is None:
            self.logger.error("Rclone failed to connect.")
            raise ValueError("Rclone failed to connect.")
        return rclone

    def _train_faiss(self, index: faiss, database: pd) -> None:
        """Train faiss before saving.

        Returns:
            None

        """
        self.faiss_trainer.index = index
        self.faiss_trainer.database = database

        self.faiss_trainer.train_faiss()

    def add_vectors(self, data: Union[List[dict], dict]) -> None:
        """Add vectors to faiss and database.

        Handles list or a single data dict to add into faiss and database.

        Args:
            data (List[dict], dict): List (or a single) dict
                of data which contains query with the query_key and vector
                with the key vector_key.

        Returns:
            None

        Raises:
            ValueError: When data dict doesn't have the vector key.

        """
        if isinstance(data, dict):
            data = [data]

        # Add ids to data list
        self._add_query_id(data=data)

        # Get duplicated ids
        duplicated_ids = self._duplicated_ids(data=data)

        # Remove already existing ids from faiss and database
        self._remove_vectors_by_id(duplicated_ids)

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

        # Add to database and Faiss (pylint: disable=W0201)
        df_to_be_append = pd.DataFrame(query_list,
                                       columns=[self.faiss_config.query_key],
                                       index=query_ids)
        df_to_be_append.index.name = self.faiss_config.id_key
        self.database = pd.concat([self.database, df_to_be_append])
        self.index.add_with_ids(np.array(vector_list, dtype=np.single),
                                np.array(query_ids, dtype='int64'))

    def remove_vectors(self, data: Union[List[dict], dict]) -> None:
        """Remove vectors from faiss and database.

        Args:
            data (List[dict], dict): List (or a single) dict
                of data which contains query with the key query_key.

        Returns:
            None

        """
        if isinstance(data, dict):
            data = [data]

        # Add ids to data list
        self._add_query_id(data=data)

        # Get duplicated ids
        duplicated_ids = self._duplicated_ids(data=data)

        # Remove already existing ids from faiss and database
        self._remove_vectors_by_id(duplicated_ids)

    def search(self, query_vector: List, num_retrieve: int=10) -> List:
        """Search Faiss.

        Args:
            query_vector (list): Input vector to search.
            num_retrieve (int): Number of results to retrive [Default: 10].

        Returns:
            list: Retrived results.

        Raises:
            ValueError: When input 'query_vec' is not of type 'List'.

        """
        self.faiss_search.index = self.index
        self.faiss_search.database = self.database

        return self.faiss_search.search(query_vector=query_vector,
                                        num_retrieve=num_retrieve)

    def save_by_copy(self, faiss_file_path: str=None,
                     database_file_path: str=None, retrain: bool=True,
                     lock: threading.Lock=None) -> None:
        """Save faiss and database by copy to file.

        Save faiss and database by creating a copy to let the faiss operate
        without interruption.

        Args:
            faiss_file_path (str): path to save faiss.
            database_file_path (str): path to save database.
            retrain (bool): Whether to retrain faiss before saving or not.
            lock(threading.Lock): To lock threads.

        Returns:
            None

        """
        if lock is None:
            index_copy = copy.deepcopy(self.index)
            database_copy = copy.deepcopy(self.database)
        else:
            with lock:
                index_copy = copy.deepcopy(self.index)
                database_copy = copy.deepcopy(self.database)

        if retrain:
            self._train_faiss(index=index_copy, database=database_copy)
            self._save(faiss_file_path=faiss_file_path,
                       database_file_path=database_file_path,
                       index=self.faiss_trainer.index,
                       database=self.faiss_trainer.database)
        else:
            self._save(faiss_file_path=faiss_file_path,
                       database_file_path=database_file_path, index=index_copy,
                       database=database_copy)

    def save(self, faiss_file_path: str=None,
             database_file_path: str=None) -> None:
        """Save faiss and database to file

        Args:
            faiss_file_path (str): path to save faiss.
            database_file_path (str): path to save database.

        Returns:
            None

        """
        self._save(faiss_file_path=faiss_file_path,
                   database_file_path=database_file_path, index=self.index,
                   database=self.database)

    def load_by_copy(self, faiss_file_path: str=None,
                     database_file_path: str=None, retrain: bool=False,
                     lock: threading.Lock=None) -> None:
        """Load faiss and database by copy from file.

        Load faiss and database by creating a copy to let the faiss operate
        without interruption.

        Args:
            faiss_file_path (str): path to load faiss from.
            database_file_path (str): path to load database from.
            retrain (bool): Whether to retrain faiss after loading or not.
            lock(threading.Lock): To lock threads.

        Returns:
            None

        """
        index_copy: faiss=None
        database_copy: pd=None

        index_copy, database_copy = self._load(
            faiss_file_path=faiss_file_path,
            database_file_path=database_file_path
        )

        if retrain:
            self._train_faiss(index=index_copy, database=database_copy)
            index_copy = self.faiss_trainer.index
            database_copy = self.faiss_trainer.database

        if lock is None:
            self.index = index_copy
            self.database = database_copy
        else:
            with lock:
                self.index = index_copy
                self.database = database_copy

    def load(self, faiss_file_path: str=None,
             database_file_path: str=None) -> None:
        """Load faiss and database to file

        Args:
            faiss_file_path (str): path to load faiss from.
            database_file_path (str): path to load database from.

        Returns:
            None

        """
        self.index, self.database = self._load(
            faiss_file_path=faiss_file_path,
            database_file_path=database_file_path
        )

    def run_save_pipeline(self, lock: threading.Lock=None) -> None:
        """Update save pipeline.

        Returns:
            None

        """
        while True:
            # Calculate relative time
            relative_time = (round(time.time()) \
                            - self.faiss_config.faiss_update_offset) \
                            % self.faiss_config.faiss_update_time

            # Check if nlist is greater than ntotal of index
            if self.index.ntotal < self.faiss_config.nlist:
                self.logger.warning(
                    "Cannot save the Faiss and database. \
                    'nlist' is smaller than ntoal")
                sleep_time = (self.faiss_config.faiss_update_time
                             - relative_time)
                time.sleep(sleep_time)
                continue

            # Perform update
            if (relative_time <= self.faiss_config.faiss_update_interval and
                self.save_flag is False):
                time_save_by_copy = time.time()
                self.logger.info(
                    "Start training and saving Faiss and database.")
                self.save_flag = True
                try:
                    self.save_by_copy(lock=lock)
                    self.logger.info(
                        f"Faiss and database saved successfuly. " \
                        f"Total time: {time.time() - time_save_by_copy} " \
                        "Total legth of Faiss: " \
                        f"{self.faiss_trainer.index.ntotal} " \
                        "Total length of Database: "\
                        f"{len(self.faiss_trainer.database)}")
                except Exception as exception:
                    self.logger.error("Error in saving faiss and database.",
                                    extra={"exc_message": exception},
                                    exc_info=exception)
            elif relative_time > self.faiss_config.faiss_update_interval:
                self.save_flag = False
                # Calculate remaining time to reset
                sleep_time = (self.faiss_config.faiss_update_time
                            - relative_time)
                self.logger.info(
                    f"Sleeping for {sleep_time} seconds until next save.")
                time.sleep(sleep_time)

    def run_load_pipeline(self, lock: threading.Lock=None) -> None:
        """Update load pipeline.

        Returns:
            None

        """
        while True:
            # Calculate relative time
            relative_time = (round(time.time()) \
                            - self.faiss_config.faiss_update_offset) \
                            % self.faiss_config.faiss_update_time

            # Perform update
            if (relative_time <= self.faiss_config.faiss_update_interval and
                self.load_flag is False):
                time_load_by_copy = time.time()
                self.logger.info(
                    "Start loading Faiss and database.")
                self.load_flag = True
                try:
                    self.load_by_copy(lock=lock)
                    self.logger.info(
                        f"Faiss and database loaded successfuly. " \
                        f"Total time: {time.time() - time_load_by_copy} " \
                        f"Total legth of Faiss: {self.index.ntotal} " \
                        f"Total length of Database: {len(self.database)}")
                except Exception as exception:
                    self.logger.error("Error in loading faiss and database.",
                                    extra={"exc_message": exception},
                                    exc_info=exception)
            elif relative_time > self.faiss_config.faiss_update_interval:
                self.load_flag = False
                # Calculate remaining time to reset
                sleep_time = (self.faiss_config.faiss_update_time
                              - relative_time)
                self.logger.info(
                    f"Sleeping for {sleep_time} seconds until next load.")
                time.sleep(sleep_time)

    def get_querries(self):
        """Return all queries saved in database.

        Returns:
            List: all queries saved inside dataset.

        """
        return self.database[self.faiss_config.query_key].tolist()

    @classmethod
    def init_faiss(cls, data: Union[List[dict], dict],
                   faiss_config: FaissConfig=None) -> None:
        """Initialize faiss and database.

        Args:
            data (List[dict], dict): List (or a single) dict
                of data which contains query with the key query_key.

        Returns:
            FaissService: Initialized faiss service.

        """
        FaissInitializer(data=data, faiss_config=faiss_config)

        return cls(faiss_config=faiss_config)
