import json
from candidate_scorer import _split_resume_sections, _extract_projects, _score_project_relevance, _load_models

# Init model
_load_models()

jd_text = "full stack developer python java javascript git"

with open('debug_real_resumes.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# I know vishv and uzair are in it. I'll pass the whole thing to scorer to see what happens
resumes = content.split("============================")[1:]
for i in range(0, len(resumes), 2):
    if i+1 >= len(resumes): break
    header = resumes[i].strip()
    text = resumes[i+1].strip()
    print(f"\nEvaluating: {header}")
    sections = _split_resume_sections(text)
    print("Sections found:", sections["_sections_found"])
    print("Project text length:", len(sections["projects"]))
    if len(sections["projects"]) > 0:
        print("First 100 chars of projects section:", repr(sections["projects"][:100]))
    
    projects = _extract_projects(text)
    print("Extracted projects count:", len(projects))
    for idx, p in enumerate(projects):
        print(f"  P{idx+1}: {p[:60]!r}...")
        
    score, count = _score_project_relevance(text, jd_text)
    print(f"JD: {jd_text}")
    print(f"Relevant count: {count}, Score: {score}")

