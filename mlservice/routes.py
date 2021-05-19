import logging
import json
from flask import current_app, request
from .mlutils import test_generation

logger = logging.getLogger('routes')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


@current_app.route('/')
def healthcheck():
    return "Hello, World!"


@current_app.route('/', methods=['POST'])
def make_inference():
    logger.info('visited inference page')
    data = request.form.to_dict(flat=True)
    logger.info(f'Input raw data {data}')
    try:
        statusCode = 200
        status = 'OK'
        response_data = test_generation(data['description'], 100)
        logger.info(response_data)
    except Exception as e:
        statusCode = 400
        status = 'Fail'
        response_data = {f'{e}'}

    response = json.dumps({
        "statusCode": statusCode,
        "status": status,
        "result": response_data
    })
    return response
