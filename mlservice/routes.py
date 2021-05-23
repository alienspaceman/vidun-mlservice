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
        response_data = current_app.config['model'].generate_text(data['description'],
                                                                  temperature=0.65,
                                                                  top_k=50,
                                                                  top_p=0.9,
                                                                  no_repeat_ngram_size=3,
                                                                  num_tokens_to_produce=200
                                                                  )
        _logger.info(f'Generation result: {response_data}')
        response_data = current_app.config['model'].postprocess_text(response_data)
        _logger.info(f'Postprocessing result: {response_data}')
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
