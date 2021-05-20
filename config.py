import dotenv
import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import os
import sys

dotenv.load_dotenv()

DIR_ROOT = pathlib.Path(__file__).resolve().parent

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = DIR_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'

MODELS_DIR = DIR_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

CACHE_DIR = DIR_ROOT / 'cache_models'
CACHE_DIR.mkdir(exist_ok=True)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler():
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when='midnight')
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(logging.WARNING)
    return file_handler


def get_logger(logger_name):
    """Get logger with prepared handlers."""

    logger = logging.getLogger(logger_name)

    logger.setLevel(logging.INFO)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False

    return logger


class Config(object):
    DEBUG = False
    TESTING = False


class ProductionConfig(Config):
    ENV = "production"
    SERVER_NAME = os.environ.get('SERVER')


class DevelopmentConfig(Config):
    ENV = "development"
    DEBUG = True
    SERVER_NAME = os.environ.get('SERVER')
    MODEL_PATH = MODELS_DIR / os.environ.get('MODEL_PATH')
    CACHE_DIR = CACHE_DIR
    MODEL_OPTIMIZED_PATH = MODELS_DIR / os.environ.get('MODEL_OPTIMIZED_PATH')
