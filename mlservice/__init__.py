import os
import dotenv
from flask import Flask
from config import get_logger
from mlservice.mlutils import ModelConfig

dotenv.load_dotenv()
_logger = get_logger(logger_name=__name__)


def create_app():
    app = Flask(__name__)
    # load configuration
    env_config = 'config.'
    env_config += {'dev': 'DevelopmentConfig', 'prod': 'ProductionConfig'}[os.environ.get('CONFIGURATION_SETUP')]
    _logger.info(f'Use {env_config} config')
    app.config.from_object(env_config)

    with app.app_context():
        from mlservice import routes
        _logger.info('Create model config object...')
        app.config['model'] = ModelConfig(app.config['MODEL_PATH'],
                                          str(app.config['CACHE_DIR']),
                                          str(app.config['MODEL_OPTIMIZED_PATH']))

        _logger.info('Application is successfully created!')

        return app
