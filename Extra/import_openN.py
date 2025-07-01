import boto3
import os

# Initialize S3 client
s3 = boto3.client('s3')

# Correct bucket name (no UUID here)
bucket_name = 'open-source-eeg-datasets'

# Prefix includes UUID and MAC address folder
prefix = 'ds005185-download/'
# Local directory to save files
local_dir = 'OpenNeuro_2019/'

# Create the local directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Paginate through all objects under prefix
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    if 'Contents' in page:
        for obj in page['Contents']:
            key = obj['Key']
            if key.endswith('/') or 'sourcedata/' in key:
                continue
            # Compute local path
            local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading {key} to {local_path}")
            s3.download_file(bucket_name, key, local_path)
    else:
        print("No files found under the specified prefix.")
