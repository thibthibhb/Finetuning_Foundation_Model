import boto3
import os

# Initialize S3 client (assumes you have AWS credentials configured)
s3 = boto3.client('s3')

# Define bucket and prefix
bucket_name = 'open-source-eeg-datasets'
prefix = 'ds004348-download/'

# Define local directory to save files
local_dir = 'OpenNeuro2017'

# Create local directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# List objects under the prefix
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

# Download each file
if 'Contents' in response:
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('/'):  # Skip folders
            continue
        local_path = os.path.join(local_dir, os.path.relpath(key, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {key} to {local_path}")
        s3.download_file(bucket_name, key, local_path)
else:
    print("No files found under the specified prefix.")