#!/bin/sh

./cloud_sql_proxy -dir=/cloudsql -instances=vidun-rus:europe-west1:dbvidun -credential_file=mlservice_key_gcp.json&
gunicorn --chdir app run:app -w 4 --threads 4 -b 0.0.0.0:5555