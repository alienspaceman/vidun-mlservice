import os


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
    MODEL_PATH = os.environ.get('MODEL_PATH')
    CACHE_DIR = os.environ.get('CACHE_DIR')
    MODEL_OPTIMIZED_PATH = os.environ.get('MODEL_OPTIMIZED_PATH')
