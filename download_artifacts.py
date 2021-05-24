import pathlib
from google.cloud import storage
from config import Config


if __name__ == '__main__':

    storage_client = storage.Client.from_service_account_json(Config.GOOGLE_APPLICATION_CREDENTIALS)
    bucket = storage_client.get_bucket(bucket_or_name=Config.GOOGLE_BUCKET)

    blobs_onnx = bucket.list_blobs(prefix=Config.GOOGLE_ONNX_MODEL)  # Get list of files
    ONNX_LOCAL_DIR = Config.MODELS_DIR / Config.GOOGLE_ONNX_MODEL
    ONNX_LOCAL_DIR.mkdir(exist_ok=True)
    for blob in blobs_onnx:
        if blob.name[-1] == '/':
            continue
        filename = blob.name.split('/')[-1]
        blob.download_to_filename(ONNX_LOCAL_DIR / filename)  # Download

    PYTORCH_LOCAL_DIR = Config.MODELS_DIR / Config.GOOGLE_PYTORCH_MODEL
    PYTORCH_LOCAL_DIR.mkdir(exist_ok=True)
    blobs_pytorch = bucket.list_blobs(prefix=Config.GOOGLE_PYTORCH_MODEL)  # Get list of files
    for blob in blobs_pytorch:
        if blob.name[-1] == '/':
            continue
        filename = blob.name.split('/')[-1]
        blob.download_to_filename(PYTORCH_LOCAL_DIR / filename)  # Download

