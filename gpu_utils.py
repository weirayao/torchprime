import os
import logging
import argparse
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
    model_safetensors_path = os.path.join(local_path, "model.safetensors")
    if os.path.exists(index_json_path) or os.path.exists(model_safetensors_path):
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

    parser = argparse.ArgumentParser(description="Download GCS checkpoints.")
    parser.add_argument('--checkpoint_dirs', nargs='+', required=True, help='List of checkpoint directories')
    parser.add_argument('--resume_checkpoints', nargs='+', required=True, help='List of resume checkpoints')
    args = parser.parse_args()

    checkpoint_dirs = args.checkpoint_dirs
    resume_checkpoints = args.resume_checkpoints

    gcs_bucket = "gs://sfr-text-diffusion-model-research/dllm_sft_checkpoints"
    local_checkpoint_path = "/export/agentstudio-family-2/haolin/dllm_sft_checkpoints"
    for checkpoint_dir in checkpoint_dirs:
        for resume_checkpoint in resume_checkpoints:
            checkpoint = os.path.join(checkpoint_dir, str(resume_checkpoint))
            print("📁 Starting to download checkpoint: ", checkpoint)
            download_gcs_checkpoint(gcs_bucket, checkpoint, local_checkpoint_path)
    print("✅ All checkpoints downloaded successfully.")
