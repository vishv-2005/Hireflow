"""Quick test to verify extraction and JSON storage work."""
import requests
import os
import json

BASE = "http://localhost:5001"

# Test 1: Upload a PDF resume with job description
print("=== Test 1: PDF Upload with Job Description ===")
with open("../test_data/john_doe_resume.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload",
        files={"files": ("john_doe_resume.pdf", f, "application/pdf")},
        data={"job_description": "python developer with machine learning experience"}
    )
print(f"Status: {r.status_code}")
data = r.json()
print(f"Message: {data.get('message', data.get('error'))}")
for c in data.get("candidates", []):
    print(f"  #{c['rank']} {c['name']} - Score: {c['score']} - Skills: {c['matched_skills']}")
print()

# Test 2: Check JSON file
print("=== Test 2: JSON File Check ===")
json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidates_data.json")
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        jdata = json.load(f)
    print(f"JSON file exists!")
    print(f"  Total batches: {jdata['total_batches']}")
    print(f"  Total resumes: {jdata['total_resumes']}")
    first_cand = jdata["batches"][0]["candidates"][0]
    print(f"  Has raw_text: {'raw_text' in first_cand}")
    print(f"  raw_text length: {len(first_cand.get('raw_text', ''))}")
    print(f"  Job description saved: {bool(jdata['batches'][0].get('job_description'))}")
else:
    print(f"JSON file NOT found at {json_path}")

# Test 3: Upload another PDF to verify JSON appends
print("\n=== Test 3: Second Upload (JSON Append Test) ===")
with open("../test_data/jane_smith_resume.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/upload",
        files={"files": ("jane_smith_resume.pdf", f, "application/pdf")},
        data={"job_description": "data analyst with sql skills"}
    )
print(f"Status: {r.status_code}")
data = r.json()
print(f"Message: {data.get('message', data.get('error'))}")
for c in data.get("candidates", []):
    print(f"  #{c['rank']} {c['name']} - Score: {c['score']} - Skills: {c['matched_skills']}")

# Verify JSON now has 2 batches
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        jdata = json.load(f)
    print(f"\nJSON after 2nd upload:")
    print(f"  Total batches: {jdata['total_batches']}")
    print(f"  Total resumes: {jdata['total_resumes']}")

# Test 4: Check /json-data endpoint
print("\n=== Test 4: /json-data API Endpoint ===")
r = requests.get(f"{BASE}/json-data")
print(f"Status: {r.status_code}")
api_data = r.json()
print(f"  Batches via API: {api_data.get('total_batches')}")
print(f"  Resumes via API: {api_data.get('total_resumes')}")

print("\n=== All tests done! ===")
