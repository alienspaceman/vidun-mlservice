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
