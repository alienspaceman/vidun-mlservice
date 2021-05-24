import dotenv
import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import os
import sys

dotenv.load_dotenv()

DIR_ROOT = pathlib.Path(__file__).resolve().parent

MODELS_DIR = DIR_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")
LOG_DIR = DIR_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'

CACHE_DIR = DIR_ROOT / 'cache_models'
CACHE_DIR.mkdir(exist_ok=True)


def seed_torch(seed=1029):
    import random
    import numpy
    import torch
    import onnxruntime
    from onnxruntime.capi._pybind_state import set_seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    onnxruntime.set_seed(seed)
    set_seed(seed)


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

    logger.setLevel({'dev': logging.INFO, 'prod': logging.WARNING}[os.environ.get('CONFIGURATION_SETUP')])

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.propagate = False

    return logger


class Config(object):
    DEBUG = False
    TESTING = False
    # SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
    SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://{}:{}@/{}?host=/cloudsql/{}:{}:{}".format(os.environ.get('GOOGLE_DB_USER'),
                                                                                               os.environ.get('GOOGLE_DB_PASSWORD'),
                                                                                               os.environ.get('GOOGLE_DB_NAME'),
                                                                                               os.environ.get('GOOGLE_DB_PROJECT_ID'),
                                                                                               os.environ.get('GOOGLE_DB_REGION'),
                                                                                               os.environ.get('GOOGLE_DB_INSTANCE_NAME'))

    SQLALCHEMY_TRACK_MODIFICATIONS = os.environ.get('SQLALCHEMY_TRACK_MODIFICATIONS')
    GOOGLE_BUCKET = os.environ.get('GOOGLE_BUCKET')
    GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    GOOGLE_ONNX_MODEL = os.environ.get('GOOGLE_ONNX_MODEL')
    GOOGLE_PYTORCH_MODEL = os.environ.get('GOOGLE_PYTORCH_MODEL')
    MODELS_DIR = MODELS_DIR
    CACHE_DIR = CACHE_DIR


class ProductionConfig(Config):
    ENV = "production"
    SERVER_NAME = os.environ.get('SERVER')


class DevelopmentConfig(Config):
    ENV = "development"
    DEBUG = True
    SERVER_NAME = os.environ.get('SERVER')
