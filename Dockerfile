FROM python:3.8-slim-buster
EXPOSE 5555
WORKDIR /usr/src/app/server

RUN apt-get update && apt-get install -y --fix-missing \
    wget \
    gcc \
    libpq-dev \
    python3-dev \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*


RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -f  https://download.pytorch.org/whl/torch_stable.html

RUN wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
RUN chmod +x cloud_sql_proxy
RUN mkdir /cloudsql; chmod +x /cloudsql

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY . .
RUN python download_artifacts.py

ENTRYPOINT ["./gunicorn.sh"]