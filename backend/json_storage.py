# json_storage.py - handles saving extracted resume data to JSON
# This JSON file serves as the primary data input for future ML model training
# and sorting/ranking logic. Every upload appends a new batch to this file.

import os
import json
from datetime import datetime

# JSON file location — stored in the backend directory
JSON_FILE_PATH = os.path.join(os.path.dirname(__file__), "candidates_data.json")


def _load_existing_data():
    """
    Load the existing JSON file if it exists.
    Returns the parsed data dict or a fresh skeleton if file doesn't exist.
    """
    if os.path.exists(JSON_FILE_PATH):
        try:
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"warning: could not read existing JSON file, starting fresh: {e}")
    
    # return a fresh skeleton
    return {
        "last_updated": None,
        "total_batches": 0,
        "total_resumes": 0,
        "batches": []
    }


def save_to_json(candidates_list, batch_id, job_description=""):
    """
    Saves a batch of scored candidates to the JSON file.
    Each batch contains all candidate data from one upload session.
    
    This function APPENDS to the existing JSON file — it never overwrites
    previous batches. This is critical because the JSON acts as training
    data for the ML model.
    
    Args:
        candidates_list: list of candidate dicts (from the scorer)
        batch_id: unique UUID string for this upload batch
        job_description: the job description used for this batch (if any)
    """
    # load existing data
    data = _load_existing_data()
    
    now = datetime.now().isoformat()
    
    # build the candidate entries for this batch
    batch_candidates = []
    for cand in candidates_list:
        candidate_entry = {
            "rank": cand.get("rank", 0),
            "name": cand.get("name", "Unknown"),
            "score": cand.get("score", 0.0),
            "matched_skills": cand.get("matched_skills", []),
            "filename": cand.get("filename", ""),
            "is_anomaly": cand.get("is_anomaly", False),
            "anomaly_reason": cand.get("anomaly_reason", ""),
            "raw_text": cand.get("raw_text", "")
        }
        batch_candidates.append(candidate_entry)
    
    # create the batch entry
    batch_entry = {
        "batch_id": batch_id,
        "uploaded_at": now,
        "job_description": job_description,
        "candidate_count": len(batch_candidates),
        "candidates": batch_candidates
    }
    
    # append to the batches list
    data["batches"].append(batch_entry)
    data["last_updated"] = now
    data["total_batches"] = len(data["batches"])
    data["total_resumes"] = sum(b["candidate_count"] for b in data["batches"])
    
    # write back to file
    try:
        with open(JSON_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"saved batch {batch_id} to {JSON_FILE_PATH} ({len(batch_candidates)} candidates)")
    except IOError as e:
        print(f"error writing JSON file: {e}")
        raise e
    
    return JSON_FILE_PATH


def load_json_data():
    """
    Load and return the full JSON data.
    Used by the API endpoint and for future model training input.
    """
    return _load_existing_data()


def get_all_resumes_for_training():
    """
    Flattens all batches and returns a list of all resume entries.
    This is the format the ML model will consume for training.
    Each entry has: name, raw_text, matched_skills, score, filename, etc.
    """
    data = _load_existing_data()
    all_resumes = []
    for batch in data.get("batches", []):
        for candidate in batch.get("candidates", []):
            # include batch context for the model
            candidate_with_context = {
                **candidate,
                "batch_id": batch["batch_id"],
                "job_description": batch.get("job_description", ""),
                "uploaded_at": batch.get("uploaded_at", "")
            }
            all_resumes.append(candidate_with_context)
    return all_resumes
