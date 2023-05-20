"""This module trains faiss."""
# TODO: Add detailed log.
# pylint: disable=E0401, E0611
import time
import gc

import faiss
import pandas as pd
import numpy as np

from logger.ve_logger import VeLogger
from config.faiss_config import FaissConfig


class FaissTrainer:
    """Faiss trainer class"""

    # Initialize logger
    logger = VeLogger()

    def __init__(self, faiss_config: FaissConfig=None) -> None:
        """Initializer of FaissTrainer class.

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
        self.index: faiss=None
        self.database: pd=None

    def train_faiss(self) -> None:
        """Train Faiss.

        Returns:
            None

        Raises:
            ValueError: If index or database attribute of object is None.

        """
        # Check arguments
        if self.index is None or self.database is None:
            self.logger.error("Index or databse is None.")
            raise ValueError("Index or databse is None.")

        # Extract vectors (pylint: disable=E1120)
        valid_query_id_list = []
        vector_list = []
        query_id_list = self.database.index.to_list()
        counter_id_not_in_faiss = 0
        for query_id in query_id_list:
            try:
                vector_list.append(self.index.reconstruct(query_id))
                valid_query_id_list.append(query_id)
            except RuntimeError:
                counter_id_not_in_faiss += 1
                self.logger.error(f"id: {query_id} does not exist in faiss.")
        self.logger.info(
            f"Number ids not found in faiss: {counter_id_not_in_faiss}")

        # Delete Faiss
        del self.index
        gc.collect()

        # Initialize Faiss
        quantizer = faiss.IndexFlatIP(self.faiss_config.vector_size)
        self.index = faiss.IndexIVFFlat(quantizer,
                                        self.faiss_config.vector_size,
                                        self.faiss_config.nlist,
                                        faiss.METRIC_INNER_PRODUCT)
        time.sleep(1)  # Sleep to let garbage collector operate.

        # Train Faiss (pylint: disable=E1120)
        self.index.train(np.array(vector_list, dtype=np.single))
        self.index.nprobe = self.faiss_config.nprobe
        self.index.add_with_ids(np.array(vector_list, dtype=np.single),
                                np.array(valid_query_id_list, dtype='int64'))
        del vector_list, query_id_list, valid_query_id_list  # remove vectors
        gc.collect()
        self.index.set_direct_map_type(faiss.DirectMap.Hashtable)
