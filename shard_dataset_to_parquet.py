#!/usr/bin/env python3
"""
Script to shard a dataset into parquet files with a specified number of rows per shard.
"""

import os
import argparse
from pathlib import Path
from datasets import load_from_disk, Dataset
import pyarrow.parquet as pq
import pyarrow as pa


def shard_dataset_to_parquet(
    dataset_path: str,
    output_dir: str,
    rows_per_shard: int = 100000,
    prefix: str = "shard"
):
    """
    Shard a dataset into parquet files with specified number of rows per shard.
    
    Args:
        dataset_path: Path to the dataset (either HuggingFace dataset directory or parquet file)
        output_dir: Directory to save the sharded parquet files
        rows_per_shard: Number of rows per shard (default: 100,000)
        prefix: Prefix for the shard filenames (default: "shard")
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {dataset_path}...")
    if os.path.isdir(dataset_path):
        # Load from HuggingFace datasets format
        dataset = load_from_disk(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset path format: {dataset_path}")
    
    total_rows = len(dataset)
    num_shards = (total_rows + rows_per_shard - 1) // rows_per_shard  # Ceiling division
    
    print(f"Dataset has {total_rows:,} rows")
    print(f"Creating {num_shards} shards with up to {rows_per_shard:,} rows each")
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * rows_per_shard
        end_idx = min(start_idx + rows_per_shard, total_rows)
        
        # Get the shard data
        shard_data = dataset.select(range(start_idx, end_idx))
        
        # Convert to PyArrow table for efficient parquet writing
        table = shard_data.data.table
        
        # Create shard filename with zero-padded index
        shard_filename = f"{prefix}_{shard_idx:06d}.parquet"
        shard_path = os.path.join(output_dir, shard_filename)
        
        # Write to parquet
        pq.write_table(table, shard_path, compression='snappy')
        
        print(f"Written shard {shard_idx + 1}/{num_shards}: {shard_filename} "
              f"({end_idx - start_idx:,} rows)")
    
    print(f"\nSharding complete! {num_shards} parquet files saved to {output_dir}")
    
    # Create a summary file
    summary_path = os.path.join(output_dir, "sharding_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Dataset sharding summary\n")
        f.write(f"========================\n")
        f.write(f"Source dataset: {dataset_path}\n")
        f.write(f"Total rows: {total_rows:,}\n")
        f.write(f"Rows per shard: {rows_per_shard:,}\n")
        f.write(f"Number of shards: {num_shards}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Shard filename pattern: {prefix}_XXXXXX.parquet\n")
        f.write(f"Compression: snappy\n")
    
    print(f"Summary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Shard a dataset into parquet files with specified rows per shard"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "output_dir",
        help="Directory to save the sharded parquet files"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100000,
        help="Number of rows per shard (default: 100,000)"
    )
    parser.add_argument(
        "--prefix",
        default="shard",
        help="Prefix for shard filenames (default: 'shard')"
    )
    
    args = parser.parse_args()
    
    shard_dataset_to_parquet(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()
