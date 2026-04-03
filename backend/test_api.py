import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath('.'))

from app import app

# Create a test client
client = app.test_client()

# Prepare an upload request with test_resumes.zip
test_zip_path = '../test_data/test_resumes.zip'

if not os.path.exists(test_zip_path):
    print(f"Test file not found: {test_zip_path}")
    sys.exit(1)

with open(test_zip_path, 'rb') as f:
    data = {
        'file': (f, 'test_resumes.zip'),
        'job_description': 'Software Engineer with experience in Python, AWS, and Machine Learning'
    }
    
    print("Sending request to /upload...")
    response = client.post('/upload', data=data, content_type='multipart/form-data')
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        json_data = response.get_json()
        print(f"Batch ID: {json_data.get('batch_id')}")
        print(f"Count: {json_data.get('count')}")
        
        candidates = json_data.get('candidates', [])
        print("\nRanked Candidates:")
        for c in candidates:
            print(f"Rank {c['rank']}: {c['name']} - Score: {c['score']} - Anomalous: {c['is_anomaly']}")
            if c['is_anomaly']:
                print(f"  Reason: {c['anomaly_reason']}")
    else:
        print(f"Error: {response.get_data(as_text=True)}")
