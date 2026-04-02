# candidate_scorer.py - scores and ranks candidates based on ML model or keyword matching
# Updated to use our Random Forest model for shortlisting predictions
# Falls back to keyword matching if model.pkl doesn't exist -- nothing breaks

import os
import re
import statistics
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# skills we look for in resume text
# kept here for fallback scoring and for displaying matched skills on dashboard
SKILL_KEYWORDS = [
    "python", "java", "javascript", "sql", "react",
    "machine learning", "data analysis", "aws", "docker", "kubernetes",
    "git", "html", "css", "node.js", "flask",
    "tensorflow", "pandas", "mongodb", "rest api", "agile"
]

# ----------------------------------------------------------------
# Load model and encoders once when the module is imported
# Using try/except so the app still runs even before training
# ----------------------------------------------------------------

# use absolute path so it works regardless of where Flask is launched from
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(_BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(_BASE_DIR, "label_encoders.pkl")

_model = None
_label_encoders = None

try:
    _model = joblib.load(MODEL_PATH)
    _label_encoders = joblib.load(ENCODERS_PATH)
    print("[candidate_scorer] Loaded model.pkl and label_encoders.pkl successfully!")
    print("[candidate_scorer] Random Forest shortlisting is ACTIVE.")
except FileNotFoundError:
    # totally fine -- just means train_model.py hasn't been run yet
    print("[candidate_scorer] model.pkl not found. Falling back to keyword scoring.")
    print("[candidate_scorer] Run 'python train_model.py' from the backend/ folder to enable ML scoring.")


def _get_matched_skills(raw_text):
    """
    Find which skills from our keyword list appear in the resume text.
    Returns a list of matched skill strings.
    """
    return [skill for skill in SKILL_KEYWORDS if skill in raw_text.lower()]


def _predict_shortlisted_score(matched_skills, raw_text):
    """
    Uses the trained Random Forest to predict a shortlisting confidence score.

    We extract features from the resume text to mimic what the model was trained on:
      - Experience_Years  : parsed from text (e.g. "5 years experience")
      - skills_count      : number of matched skills found in the resume
      - Expected_Salary   : parsed from text (e.g. "expected salary 70000")
      - Department        : guessed from keywords in the text
      - JobRole           : guessed from keywords in the text

    The model outputs class 0 or 1, and predict_proba gives us the confidence.
    We return the probability of class 1 (shortlisted) as a score out of 100.
    """
    # --- experience years ---
    # look for "5 years", "3+ years", "2 year experience" etc.
    exp_match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', raw_text, re.IGNORECASE)
    experience_years = int(exp_match.group(1)) if exp_match else 0

    # --- skills count ---
    skills_count = len(matched_skills)

    # --- expected salary ---
    # look for "salary: 70000" or "expected 80000" or just a big 5-6 digit number
    salary_match = re.search(
        r'(?:expected\s*salary|salary\s*expected|salary)[:\s]*[\$₹]?\s*(\d[\d,]+)',
        raw_text, re.IGNORECASE
    )
    if not salary_match:
        salary_match = re.search(r'[\$₹]?\s*(\d{5,6})', raw_text)
    expected_salary = int(salary_match.group(1).replace(",", "")) if salary_match else 50000

    # --- guess department from text ---
    # rough heuristic based on common keywords -- not perfect but good enough
    dept_map = {
        "IT":          ["software", "programming", "coding", "developer", "it ", "computer science"],
        "Data Science":["data science", "data analyst", "machine learning", "deep learning", "ai "],
        "HR":          ["human resources", "hr ", "talent acquisition", "recruiter", "payroll"],
        "Finance":     ["finance", "accounting", "ca ", "audit", "tax", "banking"],
        "Marketing":   ["marketing", "seo", "content", "brand", "digital marketing"],
        "Operations":  ["operations", "supply chain", "logistics", "warehouse", "procurement"],
    }
    detected_dept = "IT"  # default
    for dept, keywords in dept_map.items():
        if any(kw in raw_text.lower() for kw in keywords):
            detected_dept = dept
            break

    # --- guess job role from text ---
    role_map = {
        "Software Engineer":        ["software engineer", "software developer", "sde", "backend", "frontend"],
        "Data Analyst":             ["data analyst", "data analysis", "analyst"],
        "Data Scientist":           ["data scientist", "ml engineer", "machine learning engineer"],
        "DevOps Engineer":          ["devops", "site reliability", "sre", "cloud engineer"],
        "Network Support Engineer": ["network engineer", "networking", "cisco", "network support"],
        "HR Officer":               ["hr officer", "hr ", "human resources", "recruiter"],
        "Marketing Officer":        ["marketing officer", "marketing", "brand", "seo"],
        "Business Development Executive": ["business development", "bde", "sales"],
        "AI Engineer":              ["ai engineer", "generative ai", "llm", "nlp"],
        "Full Stack Developer (Python,React js)": ["full stack", "fullstack", "react", "django"],
    }
    detected_role = "Software Engineer"  # default
    for role, keywords in role_map.items():
        if any(kw in raw_text.lower() for kw in keywords):
            detected_role = role
            break

    # --- encode using saved label encoders ---
    le_dept = _label_encoders["Department"]
    le_role = _label_encoders["JobRole"]

    # handle unseen labels -- transform() crashes without this check
    # this took us a while to figure out
    dept_encoded = le_dept.transform([detected_dept])[0] if detected_dept in le_dept.classes_ else 0
    role_encoded = le_role.transform([detected_role])[0] if detected_role in le_role.classes_ else 0

    # --- build feature vector (must match training order exactly) ---
    # Features: Experience_Years, skills_count, Expected_Salary, Department, JobRole
    # using a DataFrame here so sklearn doesn't complain about feature names
    import pandas as pd
    features = pd.DataFrame(
        [[experience_years, skills_count, expected_salary, dept_encoded, role_encoded]],
        columns=["Experience_Years", "skills_count", "Expected_Salary", "Department", "JobRole"]
    )

    # --- predict ---
    prediction = _model.predict(features)[0]           # 0 or 1
    proba = _model.predict_proba(features)[0]           # [prob_not_shortlisted, prob_shortlisted]
    confidence_score = round(proba[1] * 100, 1)         # probability of being shortlisted, as %

    return confidence_score, bool(prediction == 1)


def score_candidate(parsed_data, job_description=None):
    """
    Main scoring function. Called for every resume.

    If the Random Forest model is loaded:
      - Uses ML to predict shortlisting confidence (0-100)
      - If a job_description is also provided, blends ML score (60%) with
        TF-IDF cosine similarity (40%) for a more targeted score

    If the model is NOT loaded (file missing):
      - Falls back to TF-IDF if job_description provided
      - Otherwise simple keyword match percentage
    """
    raw_text = parsed_data["raw_text"].lower()
    filename = parsed_data["filename"]

    # always gather matched skills -- used for display on the dashboard
    matched_skills = _get_matched_skills(raw_text)

    # -------------------------------------------------------
    # PATH A: ML scoring (Random Forest loaded)
    # -------------------------------------------------------
    if _model is not None and _label_encoders is not None:
        ml_score, is_shortlisted = _predict_shortlisted_score(matched_skills, raw_text)

        if job_description and job_description.strip():
            # blend RF score with TF-IDF similarity against job description
            # 60% ML shortlisting + 40% job description match
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([job_description.lower(), raw_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                tfidf_score = round(similarity * 100, 1)
                score = round(0.6 * ml_score + 0.4 * tfidf_score, 1)
            except ValueError:
                score = ml_score
        else:
            score = ml_score

    # -------------------------------------------------------
    # PATH B: Fallback keyword scoring (model not found)
    # -------------------------------------------------------
    else:
        if job_description and job_description.strip():
            vectorizer = TfidfVectorizer()
            try:
                tfidf_matrix = vectorizer.fit_transform([job_description.lower(), raw_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                score = round(similarity * 100, 1)
            except ValueError:
                total = len(SKILL_KEYWORDS)
                score = round((len(matched_skills) / total) * 100, 1)
        else:
            total = len(SKILL_KEYWORDS)
            score = round((len(matched_skills) / total) * 100, 1)

    name = extract_name_from_filename(filename)

    return {
        "name": name,
        "score": score,
        "matched_skills": matched_skills,
        "filename": filename,
        "raw_text": raw_text  # kept temporarily for anomaly detection
    }


def extract_name_from_filename(filename):
    """
    Extracts a human-readable name from the resume filename.
    e.g. "john_doe_resume.pdf" -> "John Doe"
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r'(?i)(resume|cv|_resume|_cv|\d+)', '', name)
    name = name.replace("_", " ").replace("-", " ")
    name = " ".join(name.split()).title().strip()
    if not name:
        name = filename
    return name


def rank_candidates(candidates_list):
    """
    Sorts candidates by score (highest first) and assigns rank numbers.
    """
    sorted_candidates = sorted(candidates_list, key=lambda x: x["score"], reverse=True)
    for i, candidate in enumerate(sorted_candidates):
        candidate["rank"] = i + 1
    return sorted_candidates


def detect_anomalies(candidates_list):
    """
    Flags resumes that are statistically too long compared to the rest of the batch.
    Anything more than 2 standard deviations above the mean length gets flagged
    as suspected keyword stuffing.
    """
    if not candidates_list:
        return candidates_list

    lengths = [len(c["raw_text"]) for c in candidates_list]
    mean_len = statistics.mean(lengths)
    std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    threshold = mean_len + (2 * std_len)

    for candidate in candidates_list:
        candidate_len = len(candidate["raw_text"])
        if std_len > 0 and candidate_len > threshold:
            candidate["is_anomaly"] = True
            candidate["anomaly_reason"] = "Suspected keyword stuffing: resume text length is statistically abnormal for this batch."
        else:
            candidate["is_anomaly"] = False
            candidate["anomaly_reason"] = ""

        candidate.pop("raw_text", None)  # clean up -- don't send huge text to frontend

    return candidates_list
