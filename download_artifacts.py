from google.cloud import storage
import config
from config import Config

storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024  # 5 MB

_logger = config.get_logger('download_artifacts')


if __name__ == '__main__':

    _logger.info('Connect to service account')
    storage_client = storage.Client.from_service_account_json(Config.GOOGLE_APPLICATION_CREDENTIALS)
    _logger.info('Connection is established')
    bucket = storage_client.get_bucket(bucket_or_name=Config.GOOGLE_BUCKET)
    _logger.info('Download files')

    PYTORCH_LOCAL_DIR = Config.MODELS_DIR / Config.GOOGLE_PYTORCH_MODEL
    PYTORCH_LOCAL_DIR.mkdir(exist_ok=True)
    _logger.info('Created local dir')
    blobs_pytorch = bucket.list_blobs(prefix=Config.GOOGLE_PYTORCH_MODEL)
    for blob in blobs_pytorch:
        if blob.name[-1] == '/':
            continue
        filename = blob.name.split('/')[-1]
        blob.download_to_filename(PYTORCH_LOCAL_DIR / filename, timeout=(600, 600))
        _logger.info(f'Downloaded {PYTORCH_LOCAL_DIR / filename}')
    _logger.info('Pytorch model is downloaded')

    blobs_onnx = bucket.list_blobs(prefix=Config.GOOGLE_ONNX_MODEL)

    ONNX_LOCAL_DIR = Config.MODELS_DIR / Config.GOOGLE_ONNX_MODEL
    ONNX_LOCAL_DIR.mkdir(exist_ok=True)
    _logger.info('Created local dir')
    for blob in blobs_onnx:
        _logger.info(f'{blob.name}')
        if blob.name[-1] == '/':
            continue
        filename = blob.name.split('/')[-1]
        blob.chunk_size = 5 * 1024 * 1024
        blob.download_to_filename(ONNX_LOCAL_DIR / filename, timeout=(1800, 1800))
        _logger.info(f'Downloaded {ONNX_LOCAL_DIR / filename}')
    _logger.info('ONNX model is downloaded')



