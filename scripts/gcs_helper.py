"""Download/upload directories from/to GCS for distributed training pods.

Usage:
    python gcs_helper.py download gs://bucket/prefix /local/path
    python gcs_helper.py upload /local/path gs://bucket/prefix
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage


def parse_gcs_path(gcs_path: str) -> tuple:
    """Parse gs://bucket/prefix into (bucket, prefix)."""
    assert gcs_path.startswith("gs://"), f"Expected gs:// path, got: {gcs_path}"
    parts = gcs_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def download(gcs_path: str, local_path: str, max_workers: int = 16):
    """Download all blobs under gcs_path to local_path."""
    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        print(f"WARNING: No files found at {gcs_path}")
        return

    print(f"Downloading {len(blobs)} files from {gcs_path} to {local_path}")

    def _download_blob(blob):
        relative = blob.name[len(prefix):].lstrip("/")
        if not relative:
            return
        dest = os.path.join(local_path, relative)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_blob, b): b for b in blobs}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == len(blobs):
                print(f"  {done}/{len(blobs)} files downloaded")

    print(f"Download complete: {local_path}")


def upload(local_path: str, gcs_path: str, max_workers: int = 16):
    """Upload local_path directory to gcs_path."""
    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files = []
    for root, _dirs, filenames in os.walk(local_path):
        for f in filenames:
            local_file = os.path.join(root, f)
            relative = os.path.relpath(local_file, local_path)
            blob_name = f"{prefix}/{relative}" if prefix else relative
            files.append((local_file, blob_name))

    if not files:
        print(f"No files to upload from {local_path}")
        return

    print(f"Uploading {len(files)} files from {local_path} to {gcs_path}")

    def _upload_file(local_file, blob_name):
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_file, lf, bn): (lf, bn) for lf, bn in files}
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 50 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files uploaded")

    print(f"Upload complete: {gcs_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python gcs_helper.py <download|upload> <src> <dst>")
        sys.exit(1)

    action = sys.argv[1]
    if action == "download":
        download(sys.argv[2], sys.argv[3])
    elif action == "upload":
        upload(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)
