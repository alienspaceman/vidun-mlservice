import os
import dotenv
from flask import Flask

dotenv.load_dotenv()


def create_app():
    app = Flask(__name__)
    # load configuration
    env_config = 'config.'
    env_config += {'dev': 'DevelopmentConfig', 'prod': 'ProductionConfig'}[os.environ.get('CONFIGURATION_SETUP')]
    app.config.from_object(env_config)

    return app
