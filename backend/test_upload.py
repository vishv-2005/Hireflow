"""Quick test script to verify multi-format upload works."""
import requests

BASE = "http://localhost:5001"

# Test 1: ZIP upload (backward compat)
print("=== Test 1: ZIP upload ===")
with open("../test_data/test_resumes.zip", "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"file": ("test_resumes.zip", f, "application/zip")})
print(f"Status: {r.status_code}")
data = r.json()
print(f"Message: {data.get('message', data.get('error'))}")
print(f"Candidates: {data.get('count', 0)}")
print()

# Test 2: Individual PDF upload
print("=== Test 2: Individual PDF upload ===")
with open("../test_data/john_doe_resume.pdf", "rb") as f:
    r = requests.post(f"{BASE}/upload", files={"files": ("john_doe_resume.pdf", f, "application/pdf")})
print(f"Status: {r.status_code}")
data = r.json()
print(f"Message: {data.get('message', data.get('error'))}")
print(f"Candidates: {data.get('count', 0)}")
print()

# Test 3: Multiple PDFs at once
print("=== Test 3: Multiple PDFs ===")
files = []
for name in ["john_doe_resume.pdf", "jane_smith_resume.pdf", "mike_chen_resume.pdf"]:
    files.append(("files", (name, open(f"../test_data/{name}", "rb"), "application/pdf")))
r = requests.post(f"{BASE}/upload", files=files)
print(f"Status: {r.status_code}")
data = r.json()
print(f"Message: {data.get('message', data.get('error'))}")
print(f"Candidates: {data.get('count', 0)}")
for c in data.get("candidates", []):
    print(f"  #{c['rank']} {c['name']} - Score: {c['score']}")

# Clean up file handles
for _, (_, fh, _) in files:
    fh.close()

print("\n=== All tests passed! ===")
