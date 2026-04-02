<<<<<<< Updated upstream
# candidate_scorer.py - scores and ranks candidates based on keyword matching
# this is our simple version of what would eventually be a real ML model

=======
# candidate_scorer.py - scores and ranks candidates using Random Forest ML Model + NLP Semantic Matching
>>>>>>> Stashed changes
import os
import re
import ast
import statistics
<<<<<<< Updated upstream
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# these are the skills we look for in resumes
# we picked common tech skills that show up in most CS job postings
=======
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ====================================================================
# SCORING CONFIGURATION (EDITABLE!)
# Adjust these weights based on what matters most for shortlisting.
# Must sum to 1.0 (or close to it).
# ====================================================================
SCORING_WEIGHTS = {
    "skills_match":       0.35,   # How closely resume skills match JD (semantic)
    "experience":         0.25,   # Years of experience
    "certificates":       0.20,   # How relevant certificates are to JD
    "contact_info":       0.10,   # Has email, phone, linkedin
    "skills_count":       0.10,   # Raw count of distinct skills
}
STRONG_THRESHOLD = 0.6  # Score >= 60% is considered a "Strong" candidate (Class 1)

# hardcoded fallback skills if no ML/JD is available
>>>>>>> Stashed changes
SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "react", "machine learning", 
    "data analysis", "aws", "docker", "kubernetes", "git", "html", 
    "css", "node.js", "flask", "tensorflow", "pandas", "mongodb", "rest api", "agile"
]

<<<<<<< Updated upstream

def score_candidate(parsed_data, job_description=None):
    """
    Takes the parsed resume data and calculates a score.
    If a job_description is provided, we use TF-IDF and cosine similarity
    (a simple ML approach) to compare the resume to the job description.
    Otherwise, we fall back to simple keyword matching.
    
    Also extracts the candidate's name from the filename.
=======
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "model.pkl")

# Lazy-loaded globals to save memory until first request
_model = None
_bert_model = None

def _load_models():
    """Loads Random Forest and BERT model into memory if not already loaded."""
    global _model, _bert_model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
            print("[candidate_scorer] Random Forest model loaded successfully!")
        except FileNotFoundError:
            print("[candidate_scorer] model.pkl not found! Will fall back to keyword scoring.")
            print("To enable ML, run: python train_model.py")
    
    if _bert_model is None and _model is not None:
        print("[candidate_scorer] Loading Sentence-BERT model...")
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[candidate_scorer] Sentence-BERT ready!")

# ============================================================
# Feature Extraction Helpers (Identical to train_model.py)
# ============================================================

def _count_skills(matched_skills):
    return len(matched_skills)

def _has_contact_info(text):
    text = str(text).lower()
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0

def _has_experience_text(text):
    text = str(text).lower()
    has_keywords = bool(re.search(r'(experience|worked at|employed at|years of)', text))
    has_dates = bool(re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now))', text))
    return 1 if (has_keywords or has_dates) else 0

def _extract_experience_years(text):
    text = str(text).lower()
    match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', text)
    if match:
        return min(int(match.group(1)), 25)
    return 0
    
def _get_matched_skills(raw_text):
    """Finds which baseline skills are explicitly in the text (for dashboard display mostly)"""
    return [s for s in SKILL_KEYWORDS if s in raw_text.lower()]

def _get_semantic_features(raw_text, job_description):
    """
    Uses Sentence-BERT to compute:
    1. Overall skill meaning vs Job Description
    2. Any found certificate relevance vs Job Description
    """
    if not job_description or not job_description.strip():
        # Fallback to software engineer baseline if none provided
        job_description = "software engineer developer"
        
    raw_text = str(raw_text).lower()
    
    # Pre-embed JD
    jd_embedding = _bert_model.encode([job_description], convert_to_tensor=True)[0]
    
    # 1. Overall Skills Semantic Score
    resume_embedding = _bert_model.encode([raw_text], convert_to_tensor=True)[0]
    sim = util.cos_sim(resume_embedding, jd_embedding).item()
    skills_match_score = min(max((sim - 0.1) * 1.5, 0.0), 1.0)
    
    # 2. Certificate Relevance Score
    cert_matches = []
    matches = re.finditer(r'((?:awscertified|certified|certification|certificate|coursera|udemy)[^\n.,]*)', raw_text)
    for m in matches:
        cert_matches.append(m.group(1).strip())
        
    cert_relevance_score = 0.0
    if cert_matches:
        cert_embeddings = _bert_model.encode(cert_matches, convert_to_tensor=True)
        cosine_scores = util.cos_sim(cert_embeddings, jd_embedding).cpu().numpy().flatten()
        relevant_scores = [s for s in cosine_scores if s > 0.3] # filter junk
        if relevant_scores:
            cert_relevance_score = min(sum(relevant_scores), 1.0)
            
    return skills_match_score, cert_relevance_score

def score_candidate(parsed_data, job_description=None):
    """
    Main scoring function. Called for every resume.
    Uses Random Forest + Sentence-BERT to give an AI confidence score (0-100%).
>>>>>>> Stashed changes
    """
    _load_models()
    
    raw_text = parsed_data["raw_text"]
    filename = parsed_data["filename"]
<<<<<<< Updated upstream

    # check which skills appear in the resume text
    # we keep this so we can still display matched skills on the dashboard
    matched_skills = []
    for skill in SKILL_KEYWORDS:
        if skill in raw_text:
            matched_skills.append(skill)

    # Use TF-IDF if we have a job description, else fallback to keyword matching
    if job_description and job_description.strip():
        # TF-IDF converts text into vectors (lists of numbers) based on word importance.
        # Cosine similarity then checks the angle between these vectors. 
        # A higher similarity means the resume closely matches the job description!
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([job_description.lower(), raw_text])
            # Getting the similarity score between the 1st item (job desc) and 2nd item (resume)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            # Convert to a percentage out of 100, rounded to 1 decimal
            score = round(similarity * 100, 1)
        except ValueError:
            # Fallback if text is completely empty and TF-IDF breaks
            total_keywords = len(SKILL_KEYWORDS)
            score = round((len(matched_skills) / total_keywords) * 100, 1)
    else:
        # calculate score as percentage - simple but it works for our prototype fallback
        total_keywords = len(SKILL_KEYWORDS)
        score = round((len(matched_skills) / total_keywords) * 100, 1)

    # try to get a name from the filename
    name = extract_name_from_filename(filename)

=======
    name = extract_name_from_filename(filename)
    
    # Baseline dashboard skills
    matched_skills = _get_matched_skills(raw_text)
    
    # -------------------------------------------------------
    # PATH A: ML pipeline is loaded
    # -------------------------------------------------------
    if _model is not None and _bert_model is not None:
        skills_count = _count_skills(matched_skills)
        has_exp = _has_experience_text(raw_text)
        has_contact = _has_contact_info(raw_text)
        exp_years = _extract_experience_years(raw_text)
        
        # NLP Semantic scores
        sm_score, cert_score = _get_semantic_features(raw_text, job_description)
        
        # Features ordered EXACTLY as trained in train_model.py
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
        
        # Predict
        proba = _model.predict_proba(X_infer)[0]        # [prob_weak, prob_strong]
        score = round(proba[1] * 100, 1)                # final out of 100
        
    # -------------------------------------------------------
    # PATH B: ML Model is missing (Fallback)
    # -------------------------------------------------------
    else:
        total = len(SKILL_KEYWORDS)
        score = round((len(matched_skills) / total) * 100, 1)
        
>>>>>>> Stashed changes
    return {
        "name": name,
        "score": score,
        "matched_skills": matched_skills,
        "filename": filename,
<<<<<<< Updated upstream
        "raw_text": raw_text  # We store this temporarily for the anomaly check later
=======
        "raw_text": raw_text  # Kept temporarily for anomaly detection
>>>>>>> Stashed changes
    }

def extract_name_from_filename(filename):
<<<<<<< Updated upstream
    """
    Pulls a candidate name out of the filename.
    We strip the extension, remove common words like 'resume' and 'cv',
    replace underscores/hyphens with spaces, and title-case it.
    It's not perfect but works for most naming conventions.
    """
    # remove the file extension
=======
    """Extracts a human-readable name from the resume filename."""
>>>>>>> Stashed changes
    name = os.path.splitext(filename)[0]

    # remove common words that aren't part of the name, and strip out numbers
    name = re.sub(r'(?i)(resume|cv|_resume|_cv|\d+)', '', name)

    # replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")

    # clean up extra spaces and title case it
    name = " ".join(name.split()).title().strip()

    # if we end up with an empty string just use the filename
    if not name:
        name = filename

    return name

def rank_candidates(candidates_list):
<<<<<<< Updated upstream
    """
    Sorts candidates by their score (highest first) and adds
    a rank number to each one. Pretty straightforward sorting.
    """
    # sort by score, highest score = rank 1
=======
    """Sorts candidates by score (highest first) and assigns rank numbers."""
>>>>>>> Stashed changes
    sorted_candidates = sorted(candidates_list, key=lambda x: x["score"], reverse=True)

    # add rank numbers starting from 1
    for i, candidate in enumerate(sorted_candidates):
        candidate["rank"] = i + 1

    return sorted_candidates

def detect_anomalies(candidates_list):
<<<<<<< Updated upstream
    """
    Runs an anomaly check on the full batch after all resumes are parsed.
    We calculate the 'skills_length' (len of raw_text) for each resume.
    If it's more than 2 standard deviations above the batch mean, 
    we flag it as an anomaly (suspected keyword stuffing!).
    """
=======
    """Flags resumes that are statistically abnormally long (keyword stuffing check)."""
>>>>>>> Stashed changes
    if not candidates_list:
        return candidates_list

    # Get raw_text lengths for the whole batch
    lengths = [len(c["raw_text"]) for c in candidates_list]
    mean_len = statistics.mean(lengths)
    
    # We need at least 2 resumes to get a standard deviation
    if len(lengths) > 1:
        std_len = statistics.stdev(lengths)
    else:
        std_len = 0.0

    threshold = mean_len + (2 * std_len)

    # Flag anything above the threshold
    for candidate in candidates_list:
        candidate_len = len(candidate["raw_text"])
        
        # Only check anomalies if there is a deviance in the batch
        if std_len > 0 and candidate_len > threshold:
            candidate["is_anomaly"] = True
            candidate["anomaly_reason"] = "Suspected keyword stuffing: resume text length is statistically abnormal for this batch."
        else:
            candidate["is_anomaly"] = False
            candidate["anomaly_reason"] = ""
<<<<<<< Updated upstream
            
        # Optional cleanup so we don't send huge strings to the frontend
        candidate.pop("raw_text", None)
=======

        candidate.pop("raw_text", None)  # Clean up
>>>>>>> Stashed changes

    return candidates_list
