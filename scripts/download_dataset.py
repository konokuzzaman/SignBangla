#!/usr/bin/env python3
"""Stream-downloads and extracts the BdSLW60 dataset directly from Kaggle into data/BdSLW60.

Streams the ZIP archive on the fly and writes files directly to disk without needing
a duplicate 25 GB zip file in storage. Fits safely within available disk space.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
import requests
from tqdm import tqdm
from stream_unzip import stream_unzip

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "BdSLW60"
DATASET_SLUG = "hasaniut/bdslw60"
USER_AGENT = "signbangla-download-dataset"  # Kaggle answers 404 to the default python-requests User-Agent


def get_kaggle_token():
    # 1. Environment variable
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token

    # 2. ~/.kaggle/access_token
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        t = token_path.read_text(encoding="utf-8").strip()
        if t:
            return t

    # 3. Legacy ~/.kaggle/kaggle.json
    json_path = Path.home() / ".kaggle" / "kaggle.json"
    if json_path.exists():
        import json
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Kaggle basic auth header can be constructed
        import base64
        creds = f"{data['username']}:{data['key']}".encode("utf-8")
        return "Basic " + base64.b64encode(creds).decode("utf-8")

    raise RuntimeError(
        "Kaggle token not found. Please ensure ~/.kaggle/access_token exists or export KAGGLE_API_TOKEN."
    )


def get_download_url(token):
    headers = {"User-Agent": USER_AGENT}
    if token.startswith("Basic "):
        headers["Authorization"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
    resp = requests.get(url, headers=headers, allow_redirects=False)
    if resp.status_code == 302 and "Location" in resp.headers:
        return resp.headers["Location"]
    elif resp.status_code == 200:
        return url
    else:
        resp.raise_for_status()


def stream_download_and_extract(target_dir=DEFAULT_OUT, max_files=None, words_filter=None):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Authenticating with Kaggle...")
    token = get_kaggle_token()
    print("Obtaining download stream URL...")
    gcs_url = get_download_url(token)

    # Get total Content-Length if available
    head_resp = requests.head(gcs_url)
    total_bytes = int(head_resp.headers.get("Content-Length", 24850783219))
    print(f"Dataset total compressed size: {total_bytes / (1024**3):.2f} GiB")
    print(f"Target directory: {target_dir}")

    stat = shutil.disk_usage(target_dir)
    avail_gib = stat.free / (1024**3)
    print(f"Available disk space: {avail_gib:.2f} GiB")

    chunk_size = 256 * 1024  # 256 KB buffer

    pbar_net = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading & Extracting")

    def chunk_generator():
        with requests.get(gcs_url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    pbar_net.update(len(chunk))
                    yield chunk

    extracted_count = 0
    skipped_count = 0
    seen_words = set()

    try:
        for file_name_b, file_size, unzipped_chunks in stream_unzip(chunk_generator()):
            name = file_name_b.decode("utf-8", errors="replace")

            # Check word filter if set (e.g. W1-2)
            if words_filter:
                folder = name.split("/")[0] if "/" in name else ""
                match = (folder in words_filter) or any(w in name for w in words_filter)
                is_meta = any(name.endswith(ext) for ext in (".json", ".csv", ".txt", ".pdf"))
                if not match and not is_meta:
                    # If we already extracted files and encounter an unrequested folder, check if we can stop
                    if (extracted_count + skipped_count) > 0 and "/" in name:
                        folder = name.split("/")[0]
                        # Check if all requested words have been seen
                        if not any(w in folder for w in words_filter):
                            all_seen = all(any(w in seen_w for seen_w in seen_words) for w in words_filter)
                            if all_seen:
                                print(f"\nAll requested words ({', '.join(words_filter)}) downloaded. Stopping early.")
                                break
                    for _ in unzipped_chunks:
                        pass
                    continue
                if match and "/" in name:
                    seen_words.add(name.split("/")[0])

            dest_path = target_dir / name

            # If directory entry in zip
            if name.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
                for _ in unzipped_chunks:
                    pass
                continue

            # Ensure parent exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if already exists and size matches
            if dest_path.exists() and dest_path.stat().st_size == file_size:
                skipped_count += 1
                for _ in unzipped_chunks:
                    pass
                continue

            # Stream chunk-by-chunk to disk
            part_path = dest_path.with_name(dest_path.name + ".part")
            with open(part_path, "wb") as f_out:
                for chunk in unzipped_chunks:
                    f_out.write(chunk)
            os.replace(part_path, dest_path)
            extracted_count += 1

            if max_files and extracted_count >= max_files:
                print(f"\nReached max files limit ({max_files}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
    finally:
        pbar_net.close()

    print(f"\nDone! Extracted: {extracted_count} files, Skipped (already downloaded): {skipped_count} files.")


def main():
    parser = argparse.ArgumentParser(description="Stream download and extract BdSLW60 directly into data/BdSLW60")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Stop after downloading N files (useful for testing)")
    parser.add_argument("--words", nargs="*", default=None, help="Filter specific word folders (e.g. W1-2 W3-4)")
    args = parser.parse_args()

    stream_download_and_extract(target_dir=args.output, max_files=args.limit, words_filter=args.words)


if __name__ == "__main__":
    main()
