"""
download_aimi_labels.py
-----------------------
Downloads labels_20250611.tsv, study_mapping_20250611.tsv, splits_20250611.tsv,
series_metadata_20250611.tsv, and image_ehr_crosswalk_20250418.csv from the
Stanford AIMI Portal.

WHY THIS SCRIPT EXISTS
----------------------
The Stanford AIMI portal (stanfordaimi.azurewebsites.net) requires interactive
browser authentication + DUA acceptance before any download. After you click
"Download" in the browser, the portal generates a time-limited Azure Blob
Storage SAS URL. Pass those URLs to this script — it handles the download,
directory creation, and integrity checks.

STEP-BY-STEP USAGE
------------------
1. Open the dataset page in your browser:
   https://stanfordaimi.azurewebsites.net/datasets/151848b9-8b31-4129-bc25-cefdf18f95d8

2. Log in and accept the Data Use Agreement.

3. Right-click the download button for each file → "Copy link address"
   (or click Download and capture the URL from your browser's download manager).

   The required files and their corresponding AIMI/Redivis reference paths are:
   - labels_20250611.tsv
     → aimi.inspect_a_multimodal_dataset_for_pulmonary_embolism_diagnosis_and_prognosis:2n96:v1_0.full:q80g/labels_20250611.tsv
   - study_mapping_20250611.tsv
     → aimi.inspect_a_multimodal_dataset_for_pulmonary_embolism_diagnosis_and_prognosis:2n96:v1_0.full:q80g/study_mapping_20250611.tsv
   - splits_20250611.tsv
     → aimi.inspect_a_multimodal_dataset_for_pulmonary_embolism_diagnosis_and_prognosis:2n96:v1_0.full:q80g/splits_20250611.tsv
   - series_metadata_20250611.tsv
     → aimi.inspect_a_multimodal_dataset_for_pulmonary_embolism_diagnosis_and_prognosis:2n96:v1_0.full:q80g/series_metadata_20250611.tsv
   - image_ehr_crosswalk_20250418.csv
     → aimi.inspect_a_multimodal_dataset_for_pulmonary_embolism_diagnosis_and_prognosis:2n96:v1_0.full:q80g/image_ehr_crosswalk_20250418.csv

4. Run:
   python download_aimi_labels.py \
       --labels-url  "<paste SAS URL for labels_20250611.tsv>" \
       --mapping-url "<paste SAS URL for study_mapping_20250611.tsv>" \
       --splits-url  "<paste SAS URL for splits_20250611.tsv>" \
       --metadata-url "<paste SAS URL for series_metadata_20250611.tsv>" \
       --crosswalk-url "<paste SAS URL for image_ehr_crosswalk_20250418.csv>"

   Optionally override the output directory (default: DATA_RAW/LABELS/):
   python download_aimi_labels.py --labels-url "..." --mapping-url "..." --splits-url "..." --metadata-url "..." --crosswalk-url "..." \
       --output-dir /path/to/custom/dir

NOTE: SAS URLs are time-limited (typically 24–72 hours). If you get an
      AuthenticationFailed or 403 error, return to the portal and regenerate.
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import hashlib

# ---------------------------------------------------------------------------
# Default output location (relative to this script's directory → project root)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)          # INSPECT_custom_data_preprocessing/
DEFAULT_OUTDIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATA_RAW", "LABELS"))

EXPECTED_FILES = {
    "labels_20250611.tsv":        "labels-url",
    "study_mapping_20250611.tsv": "mapping-url",
    "splits_20250611.tsv":        "splits-url",
    "series_metadata_20250611.tsv": "metadata-url",
    "image_ehr_crosswalk_20250418.csv": "crosswalk-url",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: str) -> None:
    """Stream-download url → dest_path with a progress indicator."""
    filename = os.path.basename(dest_path)

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            mb_done  = downloaded / 1_048_576
            mb_total = total_size  / 1_048_576
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
        else:
            print(f"\r  Downloaded {downloaded / 1_048_576:.1f} MB...", end="", flush=True)

    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    except urllib.error.HTTPError as e:
        print()   # newline after progress bar
        if e.code in (403, 401):
            print(f"\nERROR {e.code}: Authentication failed or SAS token expired.")
            print("Return to the AIMI portal, log in, and copy a fresh download link.")
        else:
            print(f"\nHTTP Error {e.code}: {e.reason}")
        raise
    except urllib.error.URLError as e:
        print(f"\nNetwork error: {e.reason}")
        raise
    print()   # newline after progress bar
    print(f"  Saved → {dest_path}")


def md5sum(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_file(filepath: str) -> None:
    """Basic sanity checks: file is non-empty and looks like a TSV or CSV."""
    size = os.path.getsize(filepath)
    if size == 0:
        raise ValueError(f"Downloaded file is empty: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n")

    is_csv = filepath.endswith(".csv")
    sep = "," if is_csv else "\t"
    sep_name = "comma" if is_csv else "tab"

    if sep not in header:
        raise ValueError(
            f"File does not appear to be {sep_name}-separated.\n"
            f"  First line: {header[:120]!r}\n"
            f"  If this looks like an HTML error page, the SAS URL may have expired."
        )

    col_count = len(header.split(sep))
    print(f"  ✓ {os.path.basename(filepath)}  |  {size / 1024:.1f} KB  |  {col_count} columns  |  MD5: {md5sum(filepath)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download INSPECT label files from Stanford AIMI Portal SAS URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--labels-url",
        metavar="URL",
        help="SAS URL for labels_20250611.tsv (copy from AIMI portal after login)",
    )
    parser.add_argument(
        "--mapping-url",
        metavar="URL",
        help="SAS URL for study_mapping_20250611.tsv (copy from AIMI portal after login)",
    )
    parser.add_argument(
        "--splits-url",
        metavar="URL",
        help="SAS URL for splits_20250611.tsv (copy from AIMI portal after login)",
    )
    parser.add_argument(
        "--metadata-url",
        metavar="URL",
        help="SAS URL for series_metadata_20250611.tsv (copy from AIMI portal after login)",
    )
    parser.add_argument(
        "--crosswalk-url",
        metavar="URL",
        help="SAS URL for image_ehr_crosswalk_20250418.csv (copy from AIMI portal after login)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTDIR,
        metavar="DIR",
        help=f"Directory to save files into (default: {DEFAULT_OUTDIR})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip download if the file already exists (useful for re-runs)",
    )

    args = parser.parse_args()

    url_map = {
        "labels_20250611.tsv":        args.labels_url,
        "study_mapping_20250611.tsv": args.mapping_url,
        "splits_20250611.tsv":        args.splits_url,
        "series_metadata_20250611.tsv": args.metadata_url,
        "image_ehr_crosswalk_20250418.csv": args.crosswalk_url,
    }

    # Prompt interactively for any missing URLs
    for filename, url in url_map.items():
        if not url:
            print(f"\nNo URL provided for {filename}.")
            print("  1. Go to: https://stanfordaimi.azurewebsites.net/datasets/151848b9-8b31-4129-bc25-cefdf18f95d8")
            print("  2. Log in and accept the DUA.")
            print("  3. Right-click the download button → Copy link address.")
            url_map[filename] = input(f"  Paste URL for {filename}: ").strip()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory for labels: {args.output_dir}\n")

    # Download each file
    errors = []
    for filename, url in url_map.items():
        if not url:
            print(f"WARNING: skipping {filename} — no URL provided.")
            continue

        # If it is the crosswalk and using default output directory, place it in DATA_PROCESSED
        if filename == "image_ehr_crosswalk_20250418.csv" and args.output_dir == DEFAULT_OUTDIR:
            dest_dir = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATA_PROCESSED"))
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, filename)
        else:
            dest = os.path.join(args.output_dir, filename)

        if args.skip_existing and os.path.exists(dest):
            print(f"  Skipping {filename} (already exists, --skip-existing)")
            validate_file(dest)
            continue

        try:
            download_file(url, dest)
            validate_file(dest)
        except Exception as e:
            errors.append((filename, str(e)))

    print()
    if errors:
        print("The following files failed to download:")
        for fname, err in errors:
            print(f"  {fname}: {err}")
        sys.exit(1)
    else:
        print("All files downloaded and validated successfully.")
        print("Place/confirm files in their expected locations:")
        for filename, url in url_map.items():
            if not url:
                continue
            if filename == "image_ehr_crosswalk_20250418.csv" and args.output_dir == DEFAULT_OUTDIR:
                dest = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "DATA_PROCESSED", filename))
            else:
                dest = os.path.join(args.output_dir, filename)
            print(f"  {dest}")


if __name__ == "__main__":
    main()
