from datetime import datetime as dt
import json
from flask import current_app, request
from config import get_logger
from mlservice.models import db
from mlservice.models import Result

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
        preproc_text = current_app.config['model'].preprocess_text(data['description'])
        raw_response_data = current_app.config['model'].generate_text(preproc_text,
                                                                  temperature=0.65,
                                                                  top_k=50,
                                                                  top_p=0.9,
                                                                  no_repeat_ngram_size=3,
                                                                  num_tokens_to_produce=100
                                                                  )
        _logger.info(f'Generation result: {raw_response_data}')
        # response_data = raw_response_data
        response_data = current_app.config['model'].postprocess_text(raw_response_data)
        _logger.info(f'Postprocessing result: {response_data}')
        _logger.info('Create new record')
        new_record = Result(request=data['description'],
                            result=raw_response_data,
                            created_at=dt.now())
        db.session.add(new_record)  # Adds new User record to database
        db.session.commit()  # Commits all changes
        _logger.info('Changes are committed to db')
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
