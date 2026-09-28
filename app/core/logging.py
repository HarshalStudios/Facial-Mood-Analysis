import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def get_logger(name: str):
    return logging.getLogger(name)
