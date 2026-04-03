# candidate_scorer.py - scores and ranks candidates using Random Forest ML Model + NLP Semantic Matching
# ============================================================================================
# This module handles the scoring of individual resumes against a Job Description.
# PATH A: If model.pkl exists -> uses Random Forest + Sentence-BERT (full ML pipeline)
# PATH B: If model.pkl is missing -> falls back to simple keyword matching
# ============================================================================================

import os
import re
import statistics
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

# Hardcoded fallback skills if no ML/JD is available
SKILL_KEYWORDS = [
    # IT & Software
    "python", "java", "javascript", "sql", "react", "machine learning", "data analysis", "aws", "docker", 
    "kubernetes", "git", "html", "css", "node.js", "flask", "tensorflow", "pandas", "mongodb", "rest api", 
    "agile", "c++", "c#", "linux", "cloud computing", "rust", "go", "typescript", "ruby", "django", "vue.js", 
    "angular", "spring boot", "postgresql", "mysql", "redis", "elasticsearch", "graphql", "microservices", 
    "ci/cd", "jenkins", "terraform", "ansible", "azure", "gcp", "data science", "deep learning", "nlp", 
    "computer vision", "cybersecurity", "penetration testing", "scrum", "jira",

    # Marketing & Sales
    "seo", "sem", "content marketing", "social media management", "b2b sales", "crm", "google analytics", 
    "email marketing", "market research", "brand management", "digital marketing", "ppc", "google ads", 
    "facebook ads", "copywriting", "public relations", "salesforce", "hubspot", "lead generation", 
    "conversion rate optimization", "a/b testing", "affiliate marketing", "influencer marketing", "event planning", 
    "product marketing", "marketing automation", "customer success", "account management", "cold calling", 
    "negotiation", "sales presentations", "business development", "market analysis", "competitive intelligence", 
    "growth hacking", "e-commerce", "shopify", "wordpress", "adobe creative suite", "graphic design", 
    "video editing", "data visualization", "tableau", "communication skills", "b2c sales", "sales strategy", 
    "key account management", "churn reduction", "onboarding", "retention strategies",

    # Finance & Accounting
    "accounting", "financial modeling", "budgeting", "forecasting", "excel", "quickbooks", "tax preparation", 
    "auditing", "risk management", "payroll", "financial analysis", "cash flow management", "general ledger", 
    "accounts payable", "accounts receivable", "reconciliation", "gaap", "ifrs", "corporate finance", 
    "investment banking", "portfolio management", "wealth management", "credit analysis", "quantitative analysis", 
    "mergers and acquisitions", "due diligence", "private equity", "venture capital", "asset management", 
    "financial reporting", "compliance", "sec reporting", "sarbanes-oxley", "erisa", "treasury", "capital budgeting", 
    "variance analysis", "cost accounting", "bookkeeping", "xero", "sap", "oracle e-business suite", "power bi", 
    "vba", "sql for finance", "data mining", "fraud detection", "anti-money laundering", "kyc", "macroeconomics",

    # Pharma & Healthcare
    "clinical trials", "fda regulations", "gmp", "glp", "quality assurance", "pharmacovigilance", "sop development", 
    "biostatistics", "drug development", "patient care", "medical billing", "healthcare administration", "emr", 
    "ehr", "epic", "cerner", "hipaa compliance", "medical terminology", "nursing", "triage", "phlebotomy", 
    "vital signs", "cpr", "bls", "acls", "infection control", "medication administration", "pharmacology", 
    "toxicology", "biochemistry", "molecular biology", "cell culture", "pcr", "elisa", "chromatography", "hplc", 
    "mass spectrometry", "medical coding", "icd-10", "cpt coding", "clinical research", "regulatory affairs", 
    "medical writing", "data management", "health informatics", "public health", "epidemiology", 
    "healthcare consulting", "telehealth", "patient scheduling",

    # Mechanical & Engineering
    "autocad", "solidworks", "cad/cam", "thermodynamics", "hvac", "robotics", "six sigma", "lean manufacturing", 
    "fluid mechanics", "project management", "mechanical design", "fea", "ansys", "matlab", "creo", "catia", 
    "mechatronics", "plc programming", "automation", "manufacturing engineering", "qa/qc", "root cause analysis", 
    "fmea", "dfm", "gd&t", "material science", "metallurgy", "machining", "cnc programming", "welding", "pneumatics", 
    "hydraulics", "supply chain management", "inventory control", "logistics", "aerospace engineering", 
    "automotive engineering", "civil engineering", "structural analysis", "electrical engineering", "circuit design", 
    "pcb design", "microcontrollers", "iot", "systems engineering", "agile hardware", "scada", "hmi", 
    "industrial engineering", "ergonomics",

    # HR & Operations
    "talent acquisition", "onboarding", "employee relations", "performance management", "supply chain", 
    "logistics", "inventory management", "procurement", "recruiting", "sourcing", "applicant tracking systems", 
    "workday", "bamboo hr", "adp", "benefits administration", "compensation", "payroll processing", "hris", 
    "organizational development", "training and development", "employee engagement", "diversity and inclusion", 
    "conflict resolution", "labor law", "osha compliance", "fmla", "workforce planning", "succession planning", 
    "change management", "operations management", "business process improvement", "six sigma green belt", 
    "lean methodologies", "kaizen", "facility management", "vendor management", "contract negotiation", 
    "strategic planning", "key performance indicators", "okrs", "project coordination", "event management", 
    "timeline management", "resource allocation", "budget tracking", "quality control", "customer service", 
    "client relations", "dispatching", "fleet management"
]

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
            print("  To enable ML scoring, run: python train_model.py")

    if _bert_model is None and _model is not None:
        print("[candidate_scorer] Loading Sentence-BERT model...")
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[candidate_scorer] Sentence-BERT ready!")


# ============================================================
# Feature Extraction Helpers (Identical logic to train_model.py)
# ============================================================

def _count_skills(matched_skills):
    """Returns the raw count of matched skills."""
    return len(matched_skills)


def _has_contact_info(text):
    """Returns 1 if the text contains an email, phone number, or LinkedIn URL."""
    text = str(text).lower()
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0


def _has_experience_text(text):
    """Returns 1 if experience-related keywords or date ranges are found."""
    text = str(text).lower()
    has_keywords = bool(re.search(r'(experience|worked at|employed at|years of)', text))
    has_dates = bool(re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now))', text))
    return 1 if (has_keywords or has_dates) else 0


def _extract_experience_years(text):
    """Extracts a numeric 'years of experience' from text or date ranges. Capped at 25."""
    text = str(text).lower()
    
    # 1. Look for explicit "X years"
    match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', text)
    if match:
        return min(int(match.group(1)), 25)
        
    # 2. Look for date ranges (e.g., 2018 - 2022)
    import datetime
    current_year = datetime.datetime.now().year
    
    total_years = 0
    date_ranges = re.finditer(r'(20[0-2][0-9])\s*[-to–]+\s*(20[0-2][0-9]|present|now|current)', text)
    
    for dr in date_ranges:
        start_year = int(dr.group(1))
        end_str = dr.group(2)
        end_year = current_year if end_str in ['present', 'now', 'current'] else int(end_str)
            
        if end_year >= start_year:
            total_years += (end_year - start_year)
            
    if total_years > 0:
        return min(total_years, 25)
        
    return 0


def _get_matched_skills(raw_text):
    """Finds which baseline skills are explicitly in the text (for dashboard display)."""
    text_lower = raw_text.lower()
    return [s for s in SKILL_KEYWORDS if s in text_lower]


def _get_semantic_features(raw_text, job_description):
    """
    Uses Sentence-BERT to compute:
    1. Overall skill/resume meaning vs Job Description  → skills_match_score (0-1)
    2. Certificate relevance vs Job Description         → cert_relevance_score (0-1)
    """
    if not job_description or not job_description.strip():
        job_description = "software engineer developer"

    raw_text = str(raw_text).lower()

    # Pre-embed JD
    jd_embedding = _bert_model.encode([job_description], convert_to_tensor=True)[0]

    # 1. Overall Skills Semantic Score
    resume_embedding = _bert_model.encode([raw_text], convert_to_tensor=True)[0]
    sim = util.cos_sim(resume_embedding, jd_embedding).item()
    # Normalize: raw BERT cosine for long texts is usually 0.1-0.8
    skills_match_score = min(max((sim - 0.1) * 1.5, 0.0), 1.0)

    # 2. Certificate Relevance Score
    cert_patterns = [
        r'((?:aws[\s\-]?certified|certified|certification|certificate|coursera|udemy|google[\s\-]?cloud|azure[\s\-]?certified)[^\n.,;]*)',
    ]
    cert_matches = []
    for pattern in cert_patterns:
        for m in re.finditer(pattern, raw_text):
            cert_text = m.group(1).strip()
            if len(cert_text) > 5:  # skip noise
                cert_matches.append(cert_text)

    cert_relevance_score = 0.0
    if cert_matches:
        try:
            cert_embeddings = _bert_model.encode(cert_matches, convert_to_tensor=True)
            cosine_scores = util.cos_sim(cert_embeddings, jd_embedding).cpu().numpy().flatten()
            # Only count certificates with meaningful relevance (> 0.3 similarity)
            relevant_scores = [float(s) for s in cosine_scores if s > 0.3]
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

        # Predict probability of being a "Strong" candidate
        proba = _model.predict_proba(X_infer)[0]       # [prob_weak, prob_strong]
        
        # Calculate raw deterministic quality score based on weights
        quality_score = (
            (sm_score * SCORING_WEIGHTS["skills_match"]) +
            (min(exp_years / 10.0, 1.0) * SCORING_WEIGHTS["experience"]) +
            (cert_score * SCORING_WEIGHTS["certificates"]) +
            (has_contact * SCORING_WEIGHTS["contact_info"]) +
            (min(skills_count / 15.0, 1.0) * SCORING_WEIGHTS["skills_count"])
        )
        
        # Blend ML confidence (30%) with deterministic score (70%)
        # This prevents scores from getting flattened to 0% if the ML model is too strict
        combined_score = (quality_score * 0.7) + (float(proba[1]) * 0.3)
        score = round(combined_score * 100, 1)

    # -------------------------------------------------------
    # PATH B: ML Model is missing (Fallback to keyword match)
    # -------------------------------------------------------
    else:
        total = len(SKILL_KEYWORDS)
        score = round((len(matched_skills) / total) * 100, 1) if total > 0 else 0.0

    return {
        "name": name,
        "score": score,
        "matched_skills": matched_skills,
        "filename": filename,
        "raw_text": raw_text  # Kept temporarily for anomaly detection
    }


# ============================================================
# Name Extraction
# ============================================================

def extract_name_from_filename(filename):
    """Extracts a human-readable name from the resume filename."""
    name = os.path.splitext(filename)[0]

    # Remove common non-name words and digits
    name = re.sub(r'(?i)(resume|cv|_resume|_cv|\d+)', '', name)

    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")

    # Clean up extra spaces and title case it
    name = " ".join(name.split()).title().strip()

    # If we end up with an empty string just use the filename
    if not name:
        name = filename

    return name


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

    # Flag anything above the threshold
    for candidate in candidates_list:
        candidate_len = len(candidate.get("raw_text", ""))

        if std_len > 0 and candidate_len > threshold:
            candidate["is_anomaly"] = True
            candidate["anomaly_reason"] = "Suspected keyword stuffing: resume text length is statistically abnormal for this batch."
        else:
            candidate["is_anomaly"] = False
            candidate["anomaly_reason"] = ""

        # Remove raw_text before sending to frontend
        candidate.pop("raw_text", None)

    return candidates_list
