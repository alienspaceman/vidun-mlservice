#!/bin/sh
gunicorn run:app -w 4 --threads 4 -b :5555