"""
0d_download_rspect_images.py
--------------------------
Downloads the raw RSPECT dataset from the public AWS S3 bucket:
arn:aws:s3:::pulmonary-embolism-detection

STEP-BY-STEP USAGE
------------------
1. Run this script:
   python Custom/0d_download_rspect_images.py
"""

import os
import sys
import argparse
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import time

# ---------------------------------------------------------------------------
# Default output location (relative to this script's directory → project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTDIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATA_RAW"))

def strip_ns(tag):
    """Strip XML namespace for easier parsing."""
    if '}' in tag:
        return tag.split('}', 1)[1]
    return tag

def list_all_blobs(base_url):
    """List all blobs in the S3 bucket using ListObjectsV2."""
    blobs = []
    continuation_token = ""
    print("Fetching file list from AWS S3 bucket...")
    
    while True:
        url = f"{base_url}?list-type=2"
        if continuation_token:
            url += f"&continuation-token={urllib.parse.quote(continuation_token)}"
            
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                
                # Strip namespaces
                for elem in root.iter():
                    elem.tag = strip_ns(elem.tag)
                
                for contents in root.findall("Contents"):
                    key = contents.find("Key").text
                    size = int(contents.find("Size").text)
                    blobs.append((key, size))
                
                is_truncated = root.find("IsTruncated")
                if is_truncated is not None and is_truncated.text == 'true':
                    continuation_token = root.find("NextContinuationToken").text
                else:
                    break
        except Exception as e:
            print(f"Error fetching file list: {e}")
            raise
            
    print(f"Total files found in dataset bucket: {len(blobs)}")
    return blobs

def download_with_resume(blob_url, dest_path, total_size):
    """Downloads a specific blob and handles resume functionality using Range requests."""
    initial_pos = 0
    mode = "wb"
    headers = {}

    if os.path.exists(dest_path):
        initial_pos = os.path.getsize(dest_path)
        if initial_pos >= total_size:
            # File is already fully downloaded
            return True  # Skipped
        
        print(f"  Resuming download from byte index {initial_pos}...")
        headers["Range"] = f"bytes={initial_pos}-"
        mode = "ab"
    else:
        print(f"  Starting download...")

    req = urllib.request.Request(blob_url, headers=headers)

    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            
            # If server ignored the Range header and returned the whole file
            if status == 200 and initial_pos > 0:
                print("  Server returned HTTP 200. Restarting download from scratch...")
                initial_pos = 0
                mode = "wb"

            # Stream download
            chunk_size = 1024 * 1024  # 1MB chunks
            downloaded = initial_pos
            
            with open(dest_path, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress percentage & size info
                    pct = min(100.0, downloaded / total_size * 100)
                    bar_len = 40
                    filled = int(bar_len * pct / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    mb_done = downloaded / 1_048_576
                    mb_total = total_size / 1_048_576
                    print(f"\r    [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
            print()
            
            # Raise exception if download was incomplete
            if downloaded < total_size:
                raise IOError(f"Incomplete download: only got {downloaded} out of {total_size} bytes.")
                
            return False  # Downloaded
            
    except urllib.error.HTTPError as e:
        if e.code == 416:
            print("\nHTTP Error 416: Range Not Satisfiable.")
            print("The file might already be fully downloaded or corrupted. Try deleting the incomplete file and restarting.")
        else:
            print(f"\nHTTP Error {e.code}: {e.reason}")
        raise
    except Exception as e:
        print(f"\nError encountered during download: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Download raw RSPECT dataset from the public AWS S3 bucket.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTDIR,
        help=f"Directory to save the downloaded file (default: {DEFAULT_OUTDIR})"
    )
    
    args = parser.parse_args()
    
    # Ensure output_dir is absolute and expanded
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
        
    base_url = "https://pulmonary-embolism-detection.s3.us-west-2.amazonaws.com"
        
    try:
        # 1. Fetch file list
        blobs = list_all_blobs(base_url)
    except Exception as e:
        print(f"Failed to retrieve file list from S3 Bucket: {e}")
        sys.exit(1)
        
    # Retry configuration for individual file downloads
    max_retries = 30
    retry_delay = 5  # seconds
    
    total_blobs = len(blobs)
    total_bytes = sum(size for _, size in blobs)
    
    print(f"\nTarget directory: {args.output_dir}")
    print(f"Found {total_blobs} files to download (Total size: {total_bytes / 1_073_741_824:.2f} GB)\n")
    
    skipped_count = 0
    downloaded_count = 0
    
    for idx, (name, size) in enumerate(blobs):
        dest_path = os.path.join(args.output_dir, name)
        
        # Build the exact blob URL
        quoted_name = urllib.parse.quote(name, safe='/')
        blob_url = f"{base_url}/{quoted_name}"
        
        print(f"[{idx + 1}/{total_blobs}] {name} ({size / 1_048_576:.2f} MB)")
        
        # Ensure output subdirectory exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Download with retry loop
        for attempt in range(1, max_retries + 1):
            try:
                skipped = download_with_resume(blob_url, dest_path, size)
                if skipped:
                    skipped_count += 1
                else:
                    downloaded_count += 1
                break
            except KeyboardInterrupt:
                print("\nDownload manually paused by user.")
                sys.exit(0)
            except Exception as e:
                print(f"\n  [Warning] Interrupted (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("\n[Error] Maximum retry attempts reached for this file. Exiting.")
                    sys.exit(1)
                    
    print("\n🎉 All downloads completed!")
    print(f"  Files skipped (already present): {skipped_count}")
    print(f"  Files downloaded: {downloaded_count}")

if __name__ == "__main__":
    main()
