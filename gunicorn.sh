#!/bin/sh

./cloud_sql_proxy -dir=/cloudsql -instances=vidun-rus:europe-west1:dbvidun -credential_file=mlservice_key_gcp.json&
gunicorn -c gunicorn.config.py run:app