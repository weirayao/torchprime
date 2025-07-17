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
        logger.info(f"✅ Checkpoint already exists locally at {local_path}")
        return local_path

    logger.info(f"⬇️ Downloading checkpoint from {gcs_path} to {local_path} ...")
    # Use gsutil to copy recursively
    os.makedirs(local_path, exist_ok=True)
    # Use -m for parallel copy, -r for recursive
    cmd = f"gsutil -m cp -r {gcs_path}/* {local_path}/"
    ret = os.system(cmd)
    logger.info(f"✅ Download complete.")
    return local_path


if __name__ == "__main__":
    # Define checkpoint directories and resume checkpoints
    CHECKPOINT_DIRS=[
        "flex_processed_v1_qw1_7b_512_split_datafix",
        "flex_processed_v1_qw1_7b_512_split_datafix_from_hf"
    ]

    RESUME_CHECKPOINTS=[
        "16000",
        "14500",
        "12000",
        "9500",
        "7000",
        "4500",
        "2500"
    ]

    gcs_bucket = "gs://sfr-text-diffusion-model-research/consolidated_checkpoints"
    local_checkpoint_path = "/export/agentstudio-family-2/haolin/consolidated_checkpoints"
    for checkpoint_dir in CHECKPOINT_DIRS:
        for resume_checkpoint in RESUME_CHECKPOINTS:
            checkpoint = os.path.join(checkpoint_dir, str(resume_checkpoint))
            print("📁 Starting to download checkpoint: ", checkpoint)
            download_gcs_checkpoint(gcs_bucket, checkpoint, local_checkpoint_path)
    print("✅ All checkpoints downloaded successfully.")
