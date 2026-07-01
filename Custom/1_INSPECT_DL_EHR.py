import os
import redivis
from dotenv import load_dotenv

# 1. Load the hidden environment variables into Python
load_dotenv()

def download_ehr_data():
    # 2. Grab the token securely from the system
    api_token = os.getenv("REDIVIS_API_TOKEN")
    
    if not api_token:
        raise ValueError("❌ Error: REDIVIS_API_TOKEN not found in the environment. Is your .env file set up?")
    
    os.environ["REDIVIS_API_TOKEN"] = api_token

    print("Initializing connection to Redivis...")
    
    dataset = redivis.organization("ShahLab").dataset("inspect_ehr")
    
    # Resolve the path relative to your home directory cleanly
    output_path = os.path.expanduser("../DATA_RAW/EHR_CSV")
    os.makedirs(output_path, exist_ok=True)
    
    try:
        print("Fetching table metadata from dataset...")
        # List all tables available in this dataset instance
        tables = dataset.list_tables()
        
        print(f"Found {len(tables)} tables. Starting download to: {output_path}\n")
        
        # Loop through each table and download it individually
        for table in tables:
            print(f" -> Downloading table: {table.name}...")
            # Sanitize the table name for local file system storage
            safe_name = table.name.lower().replace(" ", "_")
            
            table.download(
                path=os.path.join(output_path, safe_name),
                format="csv"
            )
            
        print("\n🎉 Success! All INSPECT EHR tables have been downloaded.")
        
    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")

if __name__ == "__main__":
    download_ehr_data()