FROM python:3.8-slim
EXPOSE 5555

#RUN apt-get update && apt-get install -y gcc linux-headers-amd64 \
#    && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /deploy
WORKDIR /deploy
COPY . /deploy

# Install virtual env and all dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt -f  https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["./gunicorn.sh"]