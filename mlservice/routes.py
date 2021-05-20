import json
from flask import current_app, request
from config import get_logger

_logger = get_logger(logger_name=__name__)

@current_app.route('/')
def healthcheck():
    return "Hello, World!"


@current_app.route('/inference', methods=['POST'])
def make_inference():
    _logger.info('visited inference page')
    data = request.form.to_dict(flat=True)
    _logger.info(f'Input raw data {data}')
    try:
        statusCode = 200
        status = 'OK'
        response_data = current_app.config['model'].test_generation(data['description'], 100)
        # response_data = current_app.config['model'].tokenizer.batch_encode_plus(['вступить в говно', 'гулять под дождем'], padding=True)
        # # response_data=';rlkgjm'
        _logger.info(response_data)
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
