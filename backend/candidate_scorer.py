# candidate_scorer.py - scores and ranks candidates using Random Forest ML Model + NLP Semantic Matching
# ============================================================================================
# This module handles the scoring of individual resumes against a Job Description.
# PATH A: If model.pkl exists -> uses Random Forest + Sentence-BERT (full ML pipeline)
# PATH B: If model.pkl is missing -> falls back to simple keyword matching
#
# Shared extraction logic (section splitting, project extraction, skill matching, etc.)
# lives in resume_features.py to avoid duplication with train_model.py.
# ============================================================================================

import os
import re
import hashlib
import statistics
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Import shared extraction & config from the DRY module
from resume_features import (
    SCORING_WEIGHTS,
    STRONG_THRESHOLD,
    EXPERIENCE_NORM_CAP,
    EXPERIENCE_FLOOR,
    SKILL_KEYWORDS,
    split_resume_sections,
    is_education_context,
    is_project_context,
    is_work_context,
    count_skills,
    has_contact_info,
    has_work_experience,
    extract_experience_years,
    extract_projects,
    get_matched_skills,
    get_jd_skills,
    compute_jd_overlap,
    extract_education_quality,
    extract_certificate_mentions,
    extract_name_from_filename,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "model.pkl")

_model      = None
_bert_model = None

# JD embedding cache — avoids re-encoding the same JD for every resume in a batch
_jd_cache = {"hash": None, "embedding": None}


def _load_models():
    """Loads Random Forest and BERT model into memory if not already loaded."""
    global _model, _bert_model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            print("[candidate_scorer] Random Forest model loaded successfully!")
        except FileNotFoundError:
            print("[candidate_scorer] model.pkl not found! Will fall back to keyword scoring.")
            print("  To enable ML scoring, run: python train_model.py")

    if _bert_model is None and _model is not None:
        print("[candidate_scorer] Loading Sentence-BERT model...")
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[candidate_scorer] Sentence-BERT ready!")


def _get_jd_embedding(job_description):
    """
    Returns the BERT embedding for the JD, using a cache so we only encode it once
    per unique JD string (saves ~50ms per resume in a batch).
    """
    global _jd_cache
    jd_hash = hashlib.md5(job_description.encode('utf-8')).hexdigest()

    if _jd_cache["hash"] == jd_hash and _jd_cache["embedding"] is not None:
        return _jd_cache["embedding"]

    jd_embedding = _bert_model.encode([job_description], convert_to_tensor=True)[0]
    _jd_cache["hash"] = jd_hash
    _jd_cache["embedding"] = jd_embedding
    return jd_embedding


# ============================================================
# Semantic Scoring (BERT-based)
# ============================================================

def _score_project_relevance(raw_text, job_description):
    """
    Extracts projects from the resume and scores each against the JD
    using Sentence-BERT cosine similarity.

    Returns:
        project_score (float): 0.0 - 1.0, how relevant the best projects are to the JD
        relevant_count (int): number of projects with similarity > 0.3
    """
    if _bert_model is None:
        return 0.0, 0

    projects = extract_projects(raw_text)

    if not projects:
        return 0.0, 0

    if not job_description or not job_description.strip():
        job_description = "software engineer developer"

    try:
        jd_embedding = _get_jd_embedding(job_description)
        project_embeddings = _bert_model.encode(projects, convert_to_tensor=True)

        cosine_scores = util.cos_sim(project_embeddings, jd_embedding).cpu().numpy().flatten()

        # Only count projects with meaningful relevance (> 0.3 similarity)
        RELEVANCE_THRESHOLD = 0.3
        relevant_scores = [float(s) for s in cosine_scores if s > RELEVANCE_THRESHOLD]
        relevant_count = len(relevant_scores)

        if relevant_scores:
            # Use the average of relevant scores, capped at 1.0
            avg_score = sum(relevant_scores) / len(relevant_scores)
            # Boost slightly for having multiple relevant projects
            boost = min(relevant_count * 0.05, 0.15)
            project_score = min(avg_score + boost, 1.0)
            return project_score, relevant_count

        return 0.0, 0

    except Exception as e:
        print(f"[candidate_scorer] Project scoring error: {e}")
        return 0.0, 0


def _get_semantic_features(raw_text, job_description):
    """
    Uses Sentence-BERT to compute:
    1. Overall skill/resume meaning vs Job Description  → skills_match_score (0-1)
    2. Certificate relevance vs Job Description         → cert_relevance_score (0-1)
    """
    if not job_description or not job_description.strip():
        job_description = "software engineer developer"

    raw_text = str(raw_text).lower()

    # Pre-embed JD (cached)
    jd_embedding = _get_jd_embedding(job_description)

    # 1. Overall Skills Semantic Score
    resume_embedding = _bert_model.encode([raw_text], convert_to_tensor=True)[0]

    # Overall semantic similarity (normalized)
    sim = util.cos_sim(resume_embedding, jd_embedding).item()
    # Normalize: raw BERT cosine for long texts is usually 0.1-0.8
    skills_match_score = min(max((sim - 0.1) * 1.5, 0.0), 1.0)

    # 2. Certificate Relevance Score — uses broadened extraction
    cert_matches = extract_certificate_mentions(raw_text)

    cert_relevance_score = 0.0
    if cert_matches:
        try:
            cert_embeddings = _bert_model.encode(cert_matches, convert_to_tensor=True)
            cosine_scores = util.cos_sim(cert_embeddings, jd_embedding).cpu().numpy().flatten()
            # Only count certificates with meaningful relevance (> 0.1 similarity)
            relevant_scores = [float(s) for s in cosine_scores if s > 0.1]
            if relevant_scores:
                # Sum of relevant scores, capped at 1.0
                cert_relevance_score = min(sum(relevant_scores), 1.0)
        except Exception as e:
            print(f"[candidate_scorer] Certificate scoring error: {e}")
            cert_relevance_score = 0.0

    return skills_match_score, cert_relevance_score


# ============================================================
# Main Scoring Function
# ============================================================

def score_candidate(parsed_data, job_description=None):
    """
    Main scoring function. Called for every resume.
    Uses Random Forest + Sentence-BERT to give an AI confidence score (0-100%).
    Falls back to keyword matching if model files are not available.
    """
    _load_models()

    raw_text = parsed_data["raw_text"]
    filename = parsed_data["filename"]
    name = extract_name_from_filename(filename)

    # Always compute baseline skill matches (for dashboard display)
    matched_skills = get_matched_skills(raw_text)

    # Compute JD-specific skill overlap
    jd_skills = get_jd_skills(job_description)
    jd_overlap = compute_jd_overlap(matched_skills, jd_skills)

    # Find which of the candidate's skills directly match the JD (for highlighting)
    jd_matched_skills = [s for s in matched_skills if s in set(jd_skills)]

    # Initialize variables for return values so they exist even if ML fails
    exp_years = extract_experience_years(raw_text)
    cert_score = 0.0
    project_score = 0.0
    relevant_projects = 0
    edu_score = extract_education_quality(raw_text, job_description)

    # -------------------------------------------------------
    # PATH A: ML pipeline is loaded
    # -------------------------------------------------------
    if _model is not None and _bert_model is not None:
        skills_count = count_skills(matched_skills)
        has_exp = has_work_experience(raw_text)
        has_contact = has_contact_info(raw_text)
        exp_years = extract_experience_years(raw_text)

        # NLP Semantic scores
        sm_score, cert_score = _get_semantic_features(raw_text, job_description)

        # Project relevance scoring
        project_score, relevant_projects = _score_project_relevance(raw_text, job_description)

        # Education quality
        edu_score = extract_education_quality(raw_text, job_description)

        # Features ordered EXACTLY as trained in train_model.py
        feature_cols = [
            "skills_match_score",
            "skills_count",
            "has_experience",
            "certificate_relevance",
            "has_contact",
            "experience_years",
            "project_relevance",
            "education_quality"
        ]

        X_infer = pd.DataFrame(
            [[sm_score, skills_count, has_exp, cert_score, has_contact, exp_years, project_score, edu_score]],
            columns=feature_cols
        )

        # Predict probability of being a "Strong" candidate
        try:
            proba = _model.predict_proba(X_infer)[0]       # [prob_weak, prob_strong]
            ml_signal = float(proba[1])
        except Exception as e:
            print(f"[candidate_scorer] ML prediction warning (feature mismatch?): {e}")
            print("  Falling back to deterministic scoring only.")
            ml_signal = 0.5  # neutral fallback

        # Normalise experience with floor for fresh graduates
        exp_norm = min(exp_years / EXPERIENCE_NORM_CAP, 1.0)
        if exp_years == 0 and (skills_count >= 5 or project_score > 0.3):
            exp_norm = EXPERIENCE_FLOOR  # Don't completely zero out strong fresh grads

        # Calculate deterministic quality score with rebalanced weights
        quality_score = (
            (jd_overlap * SCORING_WEIGHTS["jd_skill_overlap"]) +
            (sm_score * SCORING_WEIGHTS["skills_match"]) +
            (exp_norm * SCORING_WEIGHTS["experience"]) +
            (cert_score * SCORING_WEIGHTS["certificates"]) +
            (has_contact * SCORING_WEIGHTS["contact_info"]) +
            (min(skills_count / 15.0, 1.0) * SCORING_WEIGHTS["skills_count"]) +
            (project_score * SCORING_WEIGHTS["project_relevance"]) +
            (edu_score * SCORING_WEIGHTS["education_quality"])
        )

        # Blend: deterministic score dominates (80%), ML provides a supporting signal (20%)
        combined_score = (quality_score * 0.80) + (ml_signal * 0.20)
        score = round(combined_score * 100, 1)

    # -------------------------------------------------------
    # PATH B: ML Model is missing (Fallback to keyword match)
    # -------------------------------------------------------
    else:
        # When no ML model, still use JD overlap if available
        if jd_skills:
            score = round(jd_overlap * 100, 1)
        else:
            total = len(SKILL_KEYWORDS)
            score = round((len(matched_skills) / total) * 100, 1) if total > 0 else 0.0

    return {
        "name":           name,
        "score":          score,
        "matched_skills": matched_skills,
        "jd_matched_skills": jd_matched_skills,
        "experience_years": exp_years,
        "has_relevant_cert": bool(cert_score > 0.0),
        "project_relevance_score": round(project_score, 3),
        "relevant_projects_count": relevant_projects,
        "education_quality": round(edu_score, 2),
        "filename": filename,
        "raw_text": raw_text  # Kept temporarily for anomaly detection
    }


# ============================================================
# Ranking
# ============================================================

def rank_candidates(candidates_list):
    """Sorts candidates by score (highest first) and assigns rank numbers."""
    sorted_candidates = sorted(candidates_list, key=lambda x: x["score"], reverse=True)

    for i, candidate in enumerate(sorted_candidates):
        candidate["rank"] = i + 1
    return sorted_candidates


# ============================================================
# Anomaly Detection
# ============================================================

def detect_anomalies(candidates_list):
    """Flags resumes that are statistically abnormally long (keyword stuffing check)."""
    if not candidates_list:
        return candidates_list

    # Get raw_text lengths for the whole batch
    lengths = [len(c.get("raw_text", "")) for c in candidates_list]
    mean_len = statistics.mean(lengths)

    # We need at least 2 resumes to get a standard deviation
    if len(lengths) > 1:
        std_len = statistics.stdev(lengths)
    else:
        std_len = 0.0

    threshold = mean_len + (2 * std_len)

    for candidate in candidates_list:
        candidate_len = len(candidate.get("raw_text", ""))

        if std_len > 0 and candidate_len > threshold:
            candidate["is_anomaly"]     = True
            candidate["anomaly_reason"] = "Suspected keyword stuffing: resume text length is statistically abnormal for this batch."
        else:
            candidate["is_anomaly"]     = False
            candidate["anomaly_reason"] = ""

        # Remove raw_text before sending to frontend
        candidate.pop("raw_text", None)

    return candidates_list
