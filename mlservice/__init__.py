import os
import dotenv
from flask import Flask
from flask_migrate import Migrate
from config import get_logger, seed_torch
from mlservice.ml_app_utils import ModelConfig
from mlservice.models import db

dotenv.load_dotenv()
_logger = get_logger(logger_name=__name__)
seed_torch()  # TODO: Fix it!

migrate = Migrate()


def create_app():
    app = Flask(__name__)
    # load configuration
    env_config = 'config.'
    env_config += {'dev': 'DevelopmentConfig', 'prod': 'ProductionConfig'}[os.environ.get('CONFIGURATION_SETUP')]
    _logger.info(f'Use {env_config} config')
    app.config.from_object(env_config)
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from mlservice import routes
        _logger.info('Create model config object...')
        seed_torch()
        app.config['model'] = ModelConfig(str(app.config['MODELS_DIR'] / app.config['GOOGLE_PYTORCH_MODEL']),
                                          app.config['CACHE_DIR'].name,
                                          str(app.config['MODELS_DIR'] / app.config['GOOGLE_ONNX_MODEL'] / os.listdir(app.config['MODELS_DIR'] / app.config['GOOGLE_ONNX_MODEL'])[0])
                                          )
        db.create_all()

        _logger.info('Application is successfully created!')

        return app
