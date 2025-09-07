"""WebDataset implementation for efficient TPU training with GCS TAR files."""

import random
import logging
import os
import json
import io
import gcsfs
import torch
import torch_xla.runtime as xr
import webdataset as wds
import numpy as np

logger = logging.getLogger(__name__)


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return xr.process_index() == 0


def list_gcs_shards(gs_prefix: str):
    # If you have an explicit list, just return it.
    # Otherwise use gcsfs to expand a pattern/prefix.
    fs = gcsfs.GCSFileSystem()  # GCE default credentials
    path = gs_prefix.replace("gs://", "") + "/**/*.tar"
    urls = fs.glob(path, recursive=True)
    urls = [f"gs://{u}" if not u.startswith("gs://") else u for u in urls]
    return urls


# Decoder for numpy - needs (key, data) arguments
def numpy_decoder(key, data):
    if isinstance(data, bytes):
        return np.load(io.BytesIO(data))
    return data


def split_by_replica(urls):
    urls = [x for x in urls]
    rank, world_size = xr.process_index(), xr.process_count()
    if world_size > 1:
        return urls[rank::world_size]
    return urls


def split_by_datloader_worker(urls):
    """Split urls per worker
    Selects a subset of urls based on Torch get_worker_info.
    Used as a shard selection function in Dataset.
    replaces wds.split_by_worker"""

    urls = [url for url in urls]

    assert isinstance(urls, list)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        wid = worker_info.id
        num_workers = worker_info.num_workers
        return urls[wid::num_workers]
    else:
        return urls


def webdataset_collate_fn(batch):
    """Collate function for WebDataset samples.
    
    When using to_tuple("npy"), WebDataset returns tuples of (numpy_array,).
    DataLoader passes a list of these tuples to the collate function.
    """
    # Extract numpy arrays from tuples
    arrays = [item[0] if isinstance(item, tuple) else item for item in batch]
    
    # Stack into a batch tensor
    input_ids = torch.from_numpy(np.stack(arrays)).long()
    
    return {"input_ids": input_ids}


def make_webdataset(
    path: str,
    shard_urls: list[str] = None,
    sample_shuffle=65536,  # sample-level shuffle buffer
    checkpoint_dir: str = None,
    seed: int = 42,
):
    """
    Builds a WebDataset pipeline that:
    - shuffles shards and splits them by node and by worker
    - reads samples from tar
    - shuffles samples
    - returns individual samples for DataLoader to batch
    """
    # Pipeline definition
    random.seed(seed)
    if shard_urls is None:
        if is_main_process():
            logger.info(f"shard_urls is None, searching for all tar files in {path}")
        shard_urls = list_gcs_shards(path)
        random.shuffle(shard_urls)
    if is_main_process():
        logger.info(f"shard_urls: {shard_urls}")
        logger.info(f"number of shard_urls: {len(shard_urls)}")

    dataset = wds.WebDataset(
        shard_urls,
        nodesplitter=split_by_replica,
        workersplitter=split_by_datloader_worker,
        repeat=False,
    ).shuffle(sample_shuffle)

    if checkpoint_dir is not None and is_main_process():
        logger.info(f"Saving shard_urls.json to {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(f"{checkpoint_dir}/shard_urls.json", "w") as f:
            json.dump(shard_urls, f, indent=4)

    dataset = dataset.decode(numpy_decoder)
    dataset = dataset.to_tuple("npy")
    return dataset


# def make_gcs_webdataset(
#     tar_urls: List[str],
#     batch_size: int,
#     num_workers: int = 128,
#     seed: int = 42,
#     shuffle_buffer: int = 32768,
#     prefetch_factor: int = 2,
#     persistent_workers: bool = True,
# ) -> DataLoader:
#     """
#     Create a WebDataset DataLoader for TPU training with proper sharding and shuffling.

#     Args:
#         tar_urls: List of GCS URLs to TAR files (e.g., ["gs://bucket/file1.tar", ...])
#         batch_size: Per-worker batch size (global_batch_size / num_workers)
#         num_workers: Total number of TPU workers (default 128)
#         seed: Random seed for shuffling
#         shuffle_buffer: Number of samples to buffer for shuffling
#         prefetch_factor: Number of batches to prefetch per dataloader worker
#         persistent_workers: Keep dataloader workers alive between epochs

#     Returns:
#         DataLoader configured for efficient TPU training
#     """
#     # Get current world size
#     world_size = xr.process_count()

#     if world_size != num_workers:
#         logger.warning("Actual world size %d differs from expected %d", world_size, num_workers)
#         num_workers = world_size

#     # Ensure deterministic shuffling across workers
#     random.seed(seed)

#     # Shuffle TAR files globally (same order for all workers)
#     shuffled_urls = tar_urls.copy()
#     random.shuffle(shuffled_urls)

#     if is_main_process():
#         logger.info("Total TAR files: %d", len(shuffled_urls))
#         logger.info("First 5 TAR files: %s", shuffled_urls[:5])

#     # Create WebDataset with proper sharding
#     dataset = (
#         wds.WebDataset(
#             shuffled_urls,
#             shardshuffle=True,  # Shuffle shards (TAR files) per worker
#             nodesplitter=wds.split_by_node,  # Split shards across nodes
#             workersplitter=wds.split_by_worker,  # Split within node
#         )
#         .shuffle(shuffle_buffer, initial=shuffle_buffer // 4)  # In-memory shuffle
#         .decode("numpy")  # Decode numpy arrays
#         .to_tuple("input_ids.npy")  # Extract input_ids numpy array
#         .batched(batch_size, partial=False)  # Create batches, drop incomplete
#     )

#     # Convert tuples to proper batch format
#     def collate_fn(batch):
#         """Convert WebDataset batch format to standard PyTorch format."""
#         # batch is a list of tuples, each tuple contains (input_ids numpy array,)
#         # Convert numpy arrays to tensors and stack
#         input_ids = torch.stack([torch.from_numpy(item[0]) for item in batch])
#         return {
#             "input_ids": input_ids,
#         }

#     # Create DataLoader with optimal settings for TPU
#     dataloader = DataLoader(
#         dataset,
#         batch_size=None,  # Batching is handled by WebDataset
#         num_workers=2,  # 2-4 workers per TPU core is usually optimal
#         pin_memory=False,  # Faster CPU-TPU transfer
#         prefetch_factor=prefetch_factor,
#         persistent_workers=persistent_workers,
#         collate_fn=collate_fn,
#     )

#     return dataloader
