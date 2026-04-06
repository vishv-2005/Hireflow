"""
Debug script to see what the project extraction finds in actual resumes.
"""
import json
import os
from candidate_scorer import _split_resume_sections, _extract_projects, _score_project_relevance, _load_models

# Load a few resumes from the JSON storage
JSON_PATH = os.path.join(os.path.dirname(__file__), "candidates_data.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Get the latest batch
latest_batch = data["batches"][-1]
jd = latest_batch.get("job_description", "")
print(f"JD: {jd}\n")
print("=" * 80)

for cand in latest_batch["candidates"][:3]:
    name = cand.get("name", "Unknown")
    raw = cand.get("raw_text", "")
    
    if not raw.strip():
        print(f"\n--- {name}: NO RAW TEXT ---")
        continue
    
    print(f"\n{'='*80}")
    print(f"CANDIDATE: {name}")
    print(f"{'='*80}")
    
    # Show first 2000 chars of raw text
    print(f"\n--- RAW TEXT (first 2000 chars) ---")
    print(raw[:2000])
    print(f"\n--- END RAW TEXT ---")
    
    # Split into sections
    sections = _split_resume_sections(raw)
    print(f"\nSections found: {sections['_sections_found']}")
    
    for sec_name in ["work", "education", "projects", "skills", "other"]:
        sec_text = sections.get(sec_name, "").strip()
        if sec_text:
            print(f"\n  [{sec_name.upper()}] ({len(sec_text)} chars):")
            print(f"    {sec_text[:200]}...")
        else:
            print(f"\n  [{sec_name.upper()}] EMPTY")
    
    # Try project extraction
    projects = _extract_projects(raw)
    print(f"\nExtracted projects: {len(projects)}")
    for i, p in enumerate(projects):
        print(f"  Project {i+1}: {p.strip()[:150]}")
    
    print()
