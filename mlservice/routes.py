from flask import current_app


@current_app.route('/')
def healthcheck():
    return "Hello, World!"