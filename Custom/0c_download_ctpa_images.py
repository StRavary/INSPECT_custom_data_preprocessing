"""
0c_download_ctpa_images.py
--------------------------
Downloads the raw CTPA images dataset from the Stanford AIMI portal SAS URL 
stored in the .env file as INSPECT_CTPA_KEY, supporting download resumption if interrupted.

STEP-BY-STEP USAGE
------------------
1. Log in to the Stanford AIMI Portal:
   https://stanfordaimi.azurewebsites.net/datasets/151848b9-8b31-4129-bc25-cefdf18f95d8

2. Locate the CTPA images download button. Right-click → "Copy link address"
   (to capture the Azure Blob SAS URL).

3. Paste the URL into your project's .env file:
   INSPECT_CTPA_KEY="https://..."

4. Run this script:
   python Custom/0c_download_ctpa_images.py
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
PROJECT_ROOT   = os.path.dirname(SCRIPT_DIR)          # .../INSPECT_custom_data_preprocessing
DEFAULT_OUTDIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATA_RAW"))

def manual_load_dotenv(dotenv_path):
    """Fallback manual parser for .env files if python-dotenv is not installed."""
    if not os.path.exists(dotenv_path):
        return {}
    env_vars = {}
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            env_vars[k] = v
    return env_vars

def list_all_blobs(base_url, sas_token):
    """List all blobs in the container using the container SAS token with list permission."""
    blobs = []
    marker = ""
    print("Fetching file list from Stanford AIMI portal container...")
    
    while True:
        url = f"{base_url}?{sas_token}&restype=container&comp=list"
        if marker:
            url += f"&marker={marker}"
            
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
                
                batch_count = 0
                for blob_el in root.findall(".//Blob"):
                    name = blob_el.find("Name").text
                    size = int(blob_el.find("Properties/Content-Length").text)
                    blobs.append((name, size))
                    batch_count += 1
                
                print(f"  Retrieved list for {len(blobs)} files...")
                
                next_marker_el = root.find("NextMarker")
                if next_marker_el is not None and next_marker_el.text:
                    marker = next_marker_el.text
                else:
                    break
        except Exception as e:
            print(f"Error fetching file list: {e}")
            raise
            
    print(f"Total files found in dataset container: {len(blobs)}")
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
        elif e.code in (403, 401):
            print(f"\nHTTP Error {e.code}: Access Denied / Forbidden.")
            print("Your Azure SAS URL has expired. Please log back in to the Stanford AIMI portal to copy a new download URL.")
        else:
            print(f"\nHTTP Error {e.code}: {e.reason}")
        raise
    except Exception as e:
        print(f"\nError encountered during download: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Download raw CTPA images from Stanford AIMI Portal SAS URL stored in .env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--key-name",
        default="INSPECT_CTPA_KEY",
        help="Name of the environment variable containing the download URL (default: INSPECT_CTPA_KEY)"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTDIR,
        help=f"Directory to save the downloaded file (default: {DEFAULT_OUTDIR})"
    )
    
    args = parser.parse_args()
    
    # Prompt user interactively to confirm or change the output directory
    print(f"Default destination directory: {args.output_dir}")
    user_dir = input("Enter custom destination directory (or press Enter to use default): ").strip()
    if user_dir:
        args.output_dir = os.path.abspath(os.path.expanduser(user_dir))
    
    dotenv_path = os.path.join(PROJECT_ROOT, ".env")
    
    # Load .env file
    ctpa_key = None
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path)
        ctpa_key = os.getenv(args.key_name)
    except ImportError:
        pass
        
    if not ctpa_key:
        env_vars = manual_load_dotenv(dotenv_path)
        ctpa_key = env_vars.get(args.key_name)
        
    if not ctpa_key:
        print(f"Error: Variable '{args.key_name}' not found in .env at {dotenv_path}")
        print("Please add the Azure SAS URL to your .env file like this:")
        print(f'  {args.key_name}="https://..."')
        sys.exit(1)
        
    # Split the SAS URL into container base URL and query string SAS token
    if "?" in ctpa_key:
        base_url, sas_token = ctpa_key.split("?", 1)
    else:
        base_url = ctpa_key
        sas_token = ""
        
    if not sas_token:
        print("Error: The URL does not contain any SAS token query parameters.")
        sys.exit(1)
        
    try:
        # 1. Fetch file list
        blobs = list_all_blobs(base_url, sas_token)
    except Exception as e:
        print(f"Failed to retrieve file list from Azure Container: {e}")
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
        
        # Build the exact blob URL with SAS token appended (quote special characters like spaces)
        quoted_name = urllib.parse.quote(name, safe='/')
        blob_url = f"{base_url}/{quoted_name}?{sas_token}"
        
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
                # Check for expired token
                if isinstance(e, urllib.error.HTTPError) and e.code in (403, 401):
                    print(f"\n[Error] Access Denied / Forbidden (HTTP {e.code}).")
                    print("Your Azure SAS URL has expired. Please log back in to the Stanford AIMI portal to get a fresh URL.")
                    sys.exit(1)
                
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
