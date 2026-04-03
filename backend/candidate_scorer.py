# candidate_scorer.py - scores and ranks candidates using Random Forest + Sentence-BERT
# Merged version: Prayag's RF inference pipeline + Vishv's BERT semantic scoring
# Falls back to keyword matching if model.pkl doesn't exist -- nothing breaks

import os
import re
import statistics
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# fallback skill list -- used for display on dashboard and basic scoring
SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "react",
    "machine learning", "data analysis", "aws", "docker", "kubernetes",
    "git", "html", "css", "node.js", "flask",
    "tensorflow", "pandas", "mongodb", "rest api", "agile"
]

# ============================================================
# SCORING WEIGHTS -- must match train_model.py exactly
# ============================================================
SCORING_WEIGHTS = {
    "skills_match":   0.35,
    "experience":     0.25,
    "certificates":   0.20,
    "contact_info":   0.10,
    "skills_count":   0.10,
}
STRONG_THRESHOLD = 0.6

# ============================================================
# Load model and BERT on startup (lazy-loaded to save memory)
# ============================================================
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_BASE_DIR, "model.pkl")

_model      = None
_bert_model = None

try:
    _model = joblib.load(MODEL_PATH)
    print("[candidate_scorer] Loaded model.pkl successfully!")
    print("[candidate_scorer] Random Forest + BERT scoring is ACTIVE.")
except FileNotFoundError:
    print("[candidate_scorer] model.pkl not found. Falling back to keyword scoring.")
    print("[candidate_scorer] Run 'python train_model.py' from the backend/ folder to enable ML scoring.")


def _load_bert():
    """Lazy-load Sentence-BERT only when first needed (saves startup time)."""
    global _bert_model
    if _bert_model is None and _model is not None:
        print("[candidate_scorer] Loading Sentence-BERT model...")
        from sentence_transformers import SentenceTransformer
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[candidate_scorer] Sentence-BERT ready!")


# ============================================================
# Feature Extraction Helpers (mirror of train_model.py)
# ============================================================

def _get_matched_skills(raw_text):
    """Find which skills from our keyword list appear in the resume."""
    return [s for s in SKILL_KEYWORDS if s in raw_text.lower()]

def _has_contact_info(text):
    text = str(text).lower()
    has_email    = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone    = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0

def _has_experience_text(text):
    text = str(text).lower()
    has_keywords = bool(re.search(r'(experience|worked at|employed at|years of)', text))
    has_dates    = bool(re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now))', text))
    return 1 if (has_keywords or has_dates) else 0

def _extract_experience_years(text):
    text = str(text).lower()
    match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', text)
    return min(int(match.group(1)), 25) if match else 0

def _get_semantic_features(raw_text, job_description):
    """
    Uses Sentence-BERT to compute:
    1. Semantic similarity of the resume vs the job description
    2. Relevance of any certificates mentioned vs the job description
    Returns (skills_match_score, cert_relevance_score) both in range 0-1.
    """
    from sentence_transformers import util

    if not job_description or not job_description.strip():
        job_description = "software engineer developer"

    raw_text = str(raw_text).lower()
    jd_embedding     = _bert_model.encode([job_description], convert_to_tensor=True)[0]
    resume_embedding = _bert_model.encode([raw_text], convert_to_tensor=True)[0]

    # Overall semantic similarity (normalized)
    sim = util.cos_sim(resume_embedding, jd_embedding).item()
    skills_match_score = min(max((sim - 0.1) * 1.5, 0.0), 1.0)

    # Certificate relevance
    cert_matches = []
    for m in re.finditer(r'((?:awscertified|certified|certification|certificate|coursera|udemy)[^\n.,]*)', raw_text):
        cert_matches.append(m.group(1).strip())

    cert_relevance_score = 0.0
    if cert_matches:
        cert_embeddings = _bert_model.encode(cert_matches, convert_to_tensor=True)
        cosine_scores   = util.cos_sim(cert_embeddings, jd_embedding).cpu().numpy().flatten()
        relevant        = [s for s in cosine_scores if s > 0.3]
        if relevant:
            cert_relevance_score = min(sum(relevant), 1.0)

    return skills_match_score, cert_relevance_score


def score_candidate(parsed_data, job_description=None):
    """
    Main scoring function called for every resume.

    PATH A (ML active): Uses Random Forest with BERT semantic features.
                        Returns probability of being "shortlisted" (0-100).
    PATH B (fallback):  If model.pkl missing, uses TF-IDF similarity or
                        simple keyword match percentage.
    """
    raw_text = parsed_data["raw_text"]
    filename = parsed_data["filename"]

    # always gather matched skills -- used for dashboard display
    matched_skills = _get_matched_skills(raw_text)

    # -----------------------------------------------------------
    # PATH A: ML pipeline (RF + BERT)
    # -----------------------------------------------------------
    if _model is not None:
        _load_bert()  # lazy-load BERT if not already loaded

        skills_count = len(matched_skills)
        has_exp      = _has_experience_text(raw_text)
        has_contact  = _has_contact_info(raw_text)
        exp_years    = _extract_experience_years(raw_text)

        # get semantic scores (vs job description or generic JD)
        sm_score, cert_score = _get_semantic_features(raw_text, job_description)

        # build feature DataFrame -- must match exact column order from training
        import pandas as pd
        feature_cols = [
            "skills_match_score",
            "skills_count",
            "has_experience",
            "certificate_relevance",
            "has_contact",
            "experience_years"
        ]
        X_infer = pd.DataFrame(
            [[sm_score, skills_count, has_exp, cert_score, has_contact, exp_years]],
            columns=feature_cols
        )

        proba = _model.predict_proba(X_infer)[0]  # [prob_weak, prob_strong]
        score = round(proba[1] * 100, 1)           # probability of being shortlisted, out of 100

    # -----------------------------------------------------------
    # PATH B: Fallback (no model)
    # -----------------------------------------------------------
    else:
        if job_description and job_description.strip():
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([job_description.lower(), raw_text.lower()])
                similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                score        = round(similarity * 100, 1)
            except ValueError:
                score = round((len(matched_skills) / len(SKILL_KEYWORDS)) * 100, 1)
        else:
            score = round((len(matched_skills) / len(SKILL_KEYWORDS)) * 100, 1)

    name = extract_name_from_filename(filename)

    return {
        "name":           name,
        "score":          score,
        "matched_skills": matched_skills,
        "filename":       filename,
        "raw_text":       raw_text  # kept temporarily for anomaly detection
    }


def extract_name_from_filename(filename):
    """Extracts a human-readable name from the resume filename."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'(?i)(resume|cv|_resume|_cv|\d+)', '', name)
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).title().strip()
    return name if name else filename


def rank_candidates(candidates_list):
    """Sorts candidates by score (highest first) and assigns rank numbers."""
    sorted_candidates = sorted(candidates_list, key=lambda x: x["score"], reverse=True)
    for i, candidate in enumerate(sorted_candidates):
        candidate["rank"] = i + 1
    return sorted_candidates


def detect_anomalies(candidates_list):
    """
    Flags resumes that are statistically much longer than others in the batch.
    More than 2 standard deviations above the mean = suspected keyword stuffing.
    """
    if not candidates_list:
        return candidates_list

    lengths  = [len(c["raw_text"]) for c in candidates_list]
    mean_len = statistics.mean(lengths)
    std_len  = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    threshold = mean_len + (2 * std_len)

    for candidate in candidates_list:
        candidate_len = len(candidate["raw_text"])
        if std_len > 0 and candidate_len > threshold:
            candidate["is_anomaly"]     = True
            candidate["anomaly_reason"] = "Suspected keyword stuffing: resume text length is statistically abnormal for this batch."
        else:
            candidate["is_anomaly"]     = False
            candidate["anomaly_reason"] = ""

        candidate.pop("raw_text", None)  # clean up -- don't send huge text to frontend

    return candidates_list
