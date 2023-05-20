"""This module contains necessary configs for faiss."""
import os


class FaissConfig:
    """Necessary configs for faiss.

    Attributes:
        faiss_file_path [required] (str): Path to faiss index file.
        database_file_path [required] (str): Path to database file.
        query_key (str): Key name of queries.
        vector_key (str): Key name of vectors.
        id_key (str): Key name of ids.
        id_generator_limiter (int): Limiter to be used by id_generator.
        vector_size (int): Size of vectors.
        nlist (int): n_list property of Faiss.
        nprobe (int): n_probe property of Faiss.
        faiss_update_time (int): Update time to save faiss and database.
        faiss_update_offset (int): Offset time to save faiss and databse.
        faiss_update_interval (int): Interval time to save faiss and database.
        quoting (int): Quoting scheme in save/load database (0, 1, 2, 3).
        delimiter (str): Character to seperate columns in database csv.
        escapechar (str): Escape character to use when save/load database csv.

    """
    faiss_file_path = str(os.getenv("FAISS_FILE_PATH")) \
                      if os.getenv("FAISS_FILE_PATH") else 'resources/faiss.idx'
    database_file_path = str(os.getenv("DATABASE_FILE_PATH")) \
                         if os.getenv("DATABASE_FILE_PATH") \
                         else 'resources/database.csv'
    query_key = str(os.getenv("QUERY_KEY")) \
                if os.getenv("QUERY_KEY") else 'query'
    vector_key = str(os.getenv("VECTOR_KEY")) \
                 if os.getenv("VECTOR_KEY") else 'vector'
    id_key = str(os.getenv("ID_KEY")) \
             if os.getenv("ID_KEY") else 'ids'
    id_generator_limiter = int(os.getenv("ID_GENERATOR_LIMITER")) \
                           if os.getenv("ID_GENERATOR_LIMITER") else 7
    vector_size = int(os.getenv("VECTOR_SIZE")) \
                  if os.getenv("VECTOR_SIZE") else 256
    nlist = int(os.getenv("N_LIST")) \
            if os.getenv("N_LIST") else 153
    nprobe = int(os.getenv("N_PROBE")) \
             if os.getenv("N_PROBE") else 153
    faiss_update_offset = int(os.getenv("FAISS_UPDATE_OFFSET")) \
                          if os.getenv("FAISS_UPDATE_OFFSET") else 0
    faiss_update_time = int(os.getenv("FAISS_UPDATE_TIME")) \
                        if os.getenv("FAISS_UPDATE_TIME") else 1800
    faiss_update_interval = int(os.getenv("FAISS_UPDATE_INTERVAL")) \
                            if os.getenv("FAISS_UPDATE_INTERVAL") else 30
    quoting = int(os.getenv("QUOTING")) \
              if os.getenv("QUOTING") else 0
    delimiter = str(os.getenv("DELIMITER")) \
                if os.getenv("DELIMITER") else '|'
    escapechar = str(os.getenv("ESCAPECHAR")) \
                 if os.getenv("ESCAPECHAR") else '\\'
    keep_default_na = bool(os.getenv("KEEP_DEFAULT_NA")) \
                      if os.getenv("KEEP_DEFAULT_NA") else False
    ceph_enable = bool(os.getenv("CEPH_ENABLE")) \
                  if os.getenv("CEPH_ENABLE") else False
    ceph_config = str(os.getenv("CEPH_CONFIG")) \
                  if os.getenv("CEPH_CONFIG") else "[models]\n type = s3\n provider = Ceph\n env_auth = false\n v2_auth = true\n access_key_id = da-dev-79c2a0a87a6e1414bf99c3d0e\n secret_access_key = 6d01528d43f4a5ec0cfd3b72976a57d0\n endpoint = https://cmn-prod-rgw.kp0.mci.dev/"
    ceph_index = str(os.getenv("CEPH_INDEX")) \
                      if os.getenv("CEPH_INDEX") \
                      else 'models:/googleclone/dev'
