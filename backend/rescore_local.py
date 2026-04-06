import json
from candidate_scorer import score_candidate, _load_models

# Load model locally
_load_models()

FILE = "candidates_data.json"
try:
    with open(FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Loaded data successfully.")
    
    if not data["batches"]:
        print("No batches found.")
    else:
        # Re-score the last batch only, to quickly fix the user's dashboard
        latest_batch = data["batches"][-1]
        jd = latest_batch.get("job_description", "")
        print(f"Re-scoring the latest batch containing {len(latest_batch['candidates'])} candidates with JD...")
        
        for cand in latest_batch['candidates']:
            raw_text = cand.get("raw_text", "")
            if not raw_text: continue
            
            # Rescore
            new_scores = score_candidate(raw_text, cand.get("filename", ""), jd)
            
            # Update candidate data with new scores
            cand["score"] = new_scores["score"]
            cand["experience_years"] = new_scores["experience_years"]
            cand["project_relevance_score"] = new_scores["project_relevance_score"]
            cand["relevant_projects_count"] = new_scores["relevant_projects_count"]
            cand["matched_skills"] = new_scores["matched_skills"]
            cand["jd_matched_skills"] = new_scores["jd_matched_skills"]
            print(f"Updated {cand['name']}: Score {cand['score']}, Projects {cand['relevant_projects_count']}")
            
        # Re-rank after scoring
        latest_batch["candidates"].sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, c in enumerate(latest_batch["candidates"]):
            c["rank"] = i + 1
            
        with open(FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Done. Saved to JSON.")
except Exception as e:
    print("Error:", e)
