import os
import logging
logger = logging.getLogger(__name__)

def download_gcs_checkpoint(gcs_bucket, checkpoint, local_checkpoint_path):
    """
    Download the checkpoint directory from GCS to a local path if not already present.
    """
    gcs_path = os.path.join(gcs_bucket, checkpoint)
    local_path = os.path.join(local_checkpoint_path, checkpoint)
    # Remove trailing slash for consistency
    gcs_path = gcs_path.rstrip("/")
    local_path = local_path.rstrip("/")
    index_json_path = os.path.join(local_path, "model.safetensors.index.json")
    if os.path.exists(index_json_path):
        logger.info(f"Checkpoint already exists locally at {local_path}")
        return local_path

    logger.info(f"Downloading checkpoint from {gcs_path} to {local_path} ...")
    # Use gsutil to copy recursively
    os.makedirs(local_path, exist_ok=True)
    # Use -m for parallel copy, -r for recursive
    cmd = f"gsutil -m cp -r {gcs_path}/* {local_path}/"
    ret = os.system(cmd)
    logger.info(f"Download complete.")
    return local_path
