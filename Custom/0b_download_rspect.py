import os
import sys
import subprocess
import argparse

def check_aws_cli():
    """Check if AWS CLI is installed."""
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_rspect(destination_dir, dry_run=False):
    """
    Downloads the RSNA-STR Pulmonary Embolism CT (RSPECT) Dataset
    from the AWS Open Data Registry.

    Bucket: s3://rsna-str-pulmonary-embolism-detection/
    Region: us-east-1
    Size: ~1 TB
    Egress: Free (AWS Open Data Sponsorship Program covers egress costs)
    Registry: https://registry.opendata.aws/rsna-pulmonary-embolism-detection/
    """
    if not check_aws_cli():
        print("Error: The AWS CLI is required to download this dataset efficiently.")
        print("Please install it: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
        sys.exit(1)

    os.makedirs(destination_dir, exist_ok=True)

    # bucket name from the AWS Open Data Registry page
    s3_uri = "s3://rsna-str-pulmonary-embolism-detection/"

    # aws s3 sync supports multipart downloads, parallel threading,
    # and resuming interrupted downloads — preferred over cp --recursive.
    # --no-sign-request: public Open Data bucket requires no authentication.
    # --region us-east-1: bucket is hosted in us-east-1.
    command = [
        "aws", "s3", "sync",
        "--no-sign-request",
        "--region", "us-east-1",
        s3_uri,
        destination_dir
    ]
    
    if dry_run:
        command.append("--dryrun")
        print("Dry run mode enabled. Simulating download...")

    print(f"Starting download from {s3_uri}")
    print(f"Destination: {destination_dir}")
    print("Command:", " ".join(command))
    print("\nNote: This dataset is ~1 TB. This process may take several hours.")
    print("Egress is free under the AWS Open Data Sponsorship Program.")
    print("If interrupted, simply re-run this script to resume — sync skips already-downloaded files.\n")

    try:
        # Run the command and stream output directly to the terminal
        subprocess.run(command, check=True)
        print("\nDownload completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nDownload failed or was interrupted (Exit code {e.returncode}).")
        print("You can re-run this script to resume the download.")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nDownload manually paused. Re-run script to resume.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the RSPECT dataset (1 TB) from AWS Open Data.")
    parser.add_argument(
        "destination", 
        nargs="?", 
        default=os.path.expanduser("../DATA_RAW/RSPECT"),
        help="Local directory to save the dataset."
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Simulate the download to see what files would be transferred."
    )
    
    args = parser.parse_args()
    
    download_rspect(args.destination, dry_run=args.dry_run)
