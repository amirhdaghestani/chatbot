"""This module is used for a singleton logger"""

import logging
import logging.config
import os


class SingletonType(type):
    """
    define singleton type.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        force to create instance once.
        Returns:
            - instance
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class VeLogger(object, metaclass=SingletonType):
    """
    This class init logger once, using singleton structure and evey other time
    we instantiate it logger class we make same instance as the first one.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    logging_ini_path = os.path.join(file_path, "logging.ini")

    def __new__(cls, config_path: str = logging_ini_path,
                log_path: str = "app.log", app_name: str = "chatbot"):
        """
        init logger only first time.
        Args:
            - config_file (str): config file path.
            - log_path (str): log file path.
            - app_name (str): application name.
        Returns:
            - cls._logger: logger object.
        Raises:
            - FileNotFoundError: if .ini does not exist in given path.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError("file {} does not exists.".format(config_path))

        cls._logger = logging.getLogger(__name__)

        logging.config.fileConfig(
            config_path,
            disable_existing_loggers=False,
            defaults={
                'log_file_name': log_path})

        old_factory = logging.getLogRecordFactory()

        def record_factory(*args, **kwargs):
            """ Add arbitrary parametes to logger """
            record = old_factory(*args, **kwargs)
            record.app_name = app_name
            return record

        logging.setLogRecordFactory(record_factory)
        return cls._logger
