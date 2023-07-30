"""This module contains necessary configs for elastic."""
import os


class ElasticConfig:
    """Necessary configs for elastic.

    Attributes:
        database_file_path [required] (str): Path to database file.
        quoting (int): Quoting scheme in save/load database (0, 1, 2, 3).
        delimiter (str): Character to seperate columns in database csv.
        escapechar (str): Escape character to use when save/load database csv.

    """
    database_file_path = str(os.getenv("ELASTIC_DATABASE_FILE_PATH")) \
                         if os.getenv("ELASTIC_DATABASE_FILE_PATH") \
                         else 'resources/database.csv'
    quoting = int(os.getenv("ELASTIC_QUOTING")) \
              if os.getenv("ELASTIC_QUOTING") else 0
    delimiter = str(os.getenv("ELASTIC_DELIMITER")) \
                if os.getenv("ELASTIC_DELIMITER") else '|'
    escapechar = str(os.getenv("ELASTIC_ESCAPECHAR")) \
                 if os.getenv("ELASTIC_ESCAPECHAR") else '\\'
    keep_default_na = bool(os.getenv("ELASTIC_KEEP_DEFAULT_NA")) \
                      if os.getenv("ELASTIC_KEEP_DEFAULT_NA") else False
    index_name = str(os.getenv("ELASTIC_INDEX_NAME")) \
                 if os.getenv("ELASTIC_INDEX_NAME") else 'knowledge_base'
    query_key = str(os.getenv("ELASTIC_QUERY_KEY")) \
                if os.getenv("ELASTIC_QUERY_KEY") else 'query'
    answer_key = str(os.getenv("ELASTIC_ANSWER_KEY")) \
                if os.getenv("ELASTIC_ANSWER_KEY") else 'answer'
    query_id_key = str(os.getenv("ELASTIC_QUERY_ID_KEY")) \
                   if os.getenv("ELASTIC_QUERY_ID_KEY") else 'ids'
    host = str(os.getenv("ELASTIC_HOST")) \
           if os.getenv("ELASTIC_HOST") else 'localhost'
    port = int(os.getenv("ELASTIC_PORT")) \
           if os.getenv("ELASTIC_PORT") else 9200
    scheme = str(os.getenv("ELASTIC_SCHEME")) \
             if os.getenv("ELASTIC_SCHEME") else 'http'
    username = str(os.getenv("ELASTIC_USERNAME")) \
               if os.getenv("ELASTIC_USERNAME") else ""
    password = str(os.getenv("ELASTIC_PASSWORD")) \
               if os.getenv("ELASTIC_PASSWORD") else ""
    fuziness = str(os.getenv("ELASTIC_FUZINESS")) \
               if os.getenv("ELASTIC_FUZINESS") else 'AUTO'
