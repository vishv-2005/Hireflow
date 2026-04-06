# train_model.py - HireFlow-AI ML Training Pipeline
# ==================================================================
# Trains a Random Forest classifier on XLSX + JSON resume data.
# Uses Sentence-BERT for semantic skill/certificate matching.
# Generates: model.pkl, confusion_matrix.png, feature_importance.png
# ==================================================================

import os
import json
import re
import ast
import sys
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util

print("=" * 60)
print("  HireFlow-AI Model Training Pipeline")
print("=" * 60)

# ------------------------------------------------------------------
# SCORING CONFIG - MUST MATCH candidate_scorer.py
# ------------------------------------------------------------------
SCORING_WEIGHTS = {
    "jd_skill_overlap":   0.25,
    "skills_match":       0.15,
    "experience":         0.15,
    "certificates":       0.10,
    "contact_info":       0.10,
    "skills_count":       0.10,
    "project_relevance":  0.15,
}
STRONG_THRESHOLD = 0.6  # quality score >= 60% = shortlisted (class 1)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # one level up from backend/

XLSX_PATH = os.path.join(PROJECT_ROOT, "Super_Resume_Dataset_Rows_1_to_1000.xlsx")
JSON_PATH = os.path.join(BASE_DIR, "candidates_data.json")
MODEL_OUTPUT = os.path.join(BASE_DIR, "model.pkl")
CM_OUTPUT = os.path.join(BASE_DIR, "confusion_matrix.png")
FI_OUTPUT = os.path.join(BASE_DIR, "feature_importance.png")

# ==================================================================
# STEP 1: Load Sentence-BERT
# ==================================================================
print("\n[Step 1] Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("  BERT model loaded.")

# ==================================================================
# STEP 2: Feature Extraction Helpers
# ==================================================================

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


# ============================================================
# Section Splitting - Same logic as candidate_scorer.py
# ============================================================

_SECTION_PATTERNS = {
    "work": re.compile(
        r'^\s*(?:work\s*experience|professional\s*experience|employment\s*history|'
        r'employment|work\s*history|career\s*history|career\s*summary|'
        r'professional\s*background|job\s*experience|positions?\s*held|'
        r'relevant\s*experience|internships?\s*(?:&|and)?\s*experience|'
        r'internship|internships|work)\s*:?\s*$',
        re.IGNORECASE
    ),
    "education": re.compile(
        r'^\s*(?:education|academic\s*background|academic\s*qualifications?|'
        r'educational\s*qualifications?|academic\s*details|academic\s*profile|'
        r'qualifications?|scholastic\s*record|degrees?)\s*:?\s*$',
        re.IGNORECASE
    ),
    "projects": re.compile(
        r'^\s*(?:projects?|academic\s*projects?|personal\s*projects?|'
        r'key\s*projects?|major\s*projects?|notable\s*projects?|'
        r'capstone\s*projects?|course\s*projects?|coursework\s*projects?|'
        r'mini\s*projects?|side\s*projects?)\s*:?\s*$',
        re.IGNORECASE
    ),
    "skills": re.compile(
        r'^\s*(?:skills?|technical\s*skills?|core\s*competenc(?:ies|e)|'
        r'key\s*skills?|areas?\s*of\s*expertise|proficienc(?:ies|y)|'
        r'technologies|tools?\s*(?:&|and)?\s*technologies)\s*:?\s*$',
        re.IGNORECASE
    ),
    "certificates": re.compile(
        r'^\s*(?:certifications?|certificates?|licenses?\s*(?:&|and)?\s*certifications?|'
        r'professional\s*certifications?|training|courses)\s*:?\s*$',
        re.IGNORECASE
    ),
    "summary": re.compile(
        r'^\s*(?:summary|objective|profile|about\s*me|personal\s*statement|'
        r'career\s*objective|professional\s*summary)\s*:?\s*$',
        re.IGNORECASE
    ),
}

_EDUCATION_CONTEXT_KEYWORDS = re.compile(
    r'(?:b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|b\.?a\b|m\.?a\b|b\.?com|m\.?com|'
    r'bachelor|master|ph\.?d|diploma|degree|university|college|institute|'
    r'school|gpa|cgpa|percentage|semester|graduated|graduation|'
    r'higher\s*secondary|hsc|ssc|10th|12th|board|cbse|icse|'
    r'mba|bba|bca|mca|enrolled)',
    re.IGNORECASE
)

_PROJECT_CONTEXT_KEYWORDS = re.compile(
    r'(?:project|capstone|mini[\s\-]?project|course\s*work|coursework|'
    r'hackathon|competition|challenge|assignment|thesis|dissertation|'
    r'research\s*paper|paper\s*titled|paper\s*on)',
    re.IGNORECASE
)

_WORK_CONTEXT_KEYWORDS = re.compile(
    r'(?:worked\s*(?:at|for|with|as)|employed\s*(?:at|by)|'
    r'job\s*(?:title|role|position)|designation|company|organization|'
    r'responsibilities|role\s*(?:and|&)\s*responsibilities|'
    r'key\s*responsibilities|duties|reporting\s*to|'
    r'full[\s\-]?time|part[\s\-]?time|contract|freelance|'
    r'team\s*(?:lead|leader|manager|member|size)|managed\s*a\s*team|'
    r'pvt\.?\s*ltd|private\s*limited|inc\.?|corp\.?|llc|'
    r'technologies|solutions|services|consulting|'
    r'software\s*(?:engineer|developer)|analyst|manager|'
    r'developer|engineer|consultant|associate|executive|'
    r'intern\s+at|interned\s+at)',
    re.IGNORECASE
)


def _split_resume_sections(text):
    """Splits resume text into named sections."""
    text = str(text)
    lines = text.split('\n')
    sections = {
        "work": [], "education": [], "projects": [],
        "skills": [], "certificates": [], "summary": [], "other": [],
    }
    current_section = "other"
    section_found = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections[current_section].append(line)
            continue
        matched_section = None
        for section_name, pattern in _SECTION_PATTERNS.items():
            if pattern.match(stripped):
                matched_section = section_name
                section_found = True
                break
        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(line)
    result = {k: '\n'.join(v) for k, v in sections.items()}
    result["_sections_found"] = section_found
    return result


def _is_education_context(surrounding_text):
    return bool(_EDUCATION_CONTEXT_KEYWORDS.search(surrounding_text))

def _is_project_context(surrounding_text):
    return bool(_PROJECT_CONTEXT_KEYWORDS.search(surrounding_text))

def _is_work_context(surrounding_text):
    return bool(_WORK_CONTEXT_KEYWORDS.search(surrounding_text))


def count_skills(skills_data):
    """Count distinct skills from text based on SKILL_KEYWORDS."""
    if pd.isna(skills_data):
        return 0
    text = str(skills_data).lower()
    count = 0
    for s in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(s) + r'\b'
        if re.search(pattern, text):
            count += 1
    return count


def has_contact_info(text):
    """Returns 1 if text contains email, phone, or LinkedIn."""
    text = str(text).lower()
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0


def has_work_experience(text):
    """Returns 1 if ACTUAL work experience is found (not education or project timelines)."""
    text = str(text)
    sections = _split_resume_sections(text)
    work_text = sections.get("work", "").lower()
    
    if sections["_sections_found"] and work_text.strip():
        has_keywords = bool(re.search(
            r'(worked at|employed at|years of experience|work experience|'
            r'professional experience|job role|designation|responsibilities)',
            work_text
        ))
        has_dates = bool(re.search(
            r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now|current))',
            work_text
        ))
        return 1 if (has_keywords or has_dates) else 0
    
    text_lower = text.lower()
    lines = text_lower.split('\n')
    for i, line in enumerate(lines):
        if re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now|current))', line):
            context_start = max(0, i - 3)
            context_end = min(len(lines), i + 4)
            context = ' '.join(lines[context_start:context_end])
            if _is_education_context(context) or _is_project_context(context):
                continue
            if _is_work_context(context):
                return 1
            if not _is_education_context(context) and not _is_project_context(context):
                if re.search(r'(pvt|ltd|inc|corp|llc|company|firm|technologies|solutions|services)', context):
                    return 1
    
    if re.search(r'\d+\s*\+?\s*years?\s+of\s+(?:work\s+)?experience', text_lower):
        return 1
    if re.search(r'(worked at|employed at|working at|employment history)', text_lower):
        return 1
    
    return 0


def extract_experience_years(text):
    """Extracts years of WORK experience only. Ignores education and project timelines."""
    text = str(text)
    current_year = datetime.datetime.now().year
    sections = _split_resume_sections(text)
    work_text = sections.get("work", "")
    
    if sections["_sections_found"] and work_text.strip():
        work_lower = work_text.lower()
        match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(work\s+)?(experience)?', work_lower)
        if match:
            return min(int(match.group(1)), 25)
        total_years = 0
        date_ranges = re.finditer(
            r'(20[0-2][0-9])\s*[-to–]+\s*(20[0-2][0-9]|present|now|current)', work_lower)
        for dr in date_ranges:
            start_year = int(dr.group(1))
            end_str = dr.group(2)
            end_year = current_year if end_str in ['present', 'now', 'current'] else int(end_str)
            if end_year >= start_year:
                total_years += (end_year - start_year)
        if total_years > 0:
            return min(total_years, 25)
        return 0
    
    text_lower = text.lower()
    match = re.search(r'(\d+)\s*\+?\s*years?\s+of\s+(?:work\s+)?experience', text_lower)
    if match:
        return min(int(match.group(1)), 25)
    
    lines = text_lower.split('\n')
    total_years = 0
    for i, line in enumerate(lines):
        date_ranges = list(re.finditer(
            r'(20[0-2][0-9])\s*[-to–]+\s*(20[0-2][0-9]|present|now|current)', line))
        if not date_ranges:
            continue
        context_start = max(0, i - 3)
        context_end = min(len(lines), i + 4)
        context = ' '.join(lines[context_start:context_end])
        if _is_education_context(context) or _is_project_context(context):
            continue
        for dr in date_ranges:
            start_year = int(dr.group(1))
            end_str = dr.group(2)
            end_year = current_year if end_str in ['present', 'now', 'current'] else int(end_str)
            if end_year >= start_year:
                total_years += (end_year - start_year)
    
    if total_years > 0:
        return min(total_years, 25)
    return 0


def _extract_projects(text):
    """Extracts individual project descriptions from the resume."""
    text = str(text)
    sections = _split_resume_sections(text)
    project_text = sections.get("projects", "")
    
    if sections["_sections_found"] and project_text.strip():
        return _split_individual_projects(project_text)
    
    projects = []
    lines = text.split('\n')
    in_project = False
    current_project = []
    for line in lines:
        stripped = line.strip().lower()
        if re.match(r'(?:project\s*(?:title|name)?\s*[:–\-])', stripped):
            if current_project:
                projects.append('\n'.join(current_project))
            current_project = [line]
            in_project = True
        elif in_project:
            is_heading = False
            for pattern in _SECTION_PATTERNS.values():
                if pattern.match(line.strip()):
                    is_heading = True
                    break
            if is_heading:
                if current_project:
                    projects.append('\n'.join(current_project))
                    current_project = []
                in_project = False
            else:
                current_project.append(line)
    if current_project:
        projects.append('\n'.join(current_project))
    return projects


def _split_individual_projects(project_section_text):
    """Splits a project section into individual projects."""
    lines = project_section_text.split('\n')
    projects = []
    current_project = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_project:
                proj_text = '\n'.join(current_project)
                if len(proj_text.strip()) > 15:
                    projects.append(proj_text)
                current_project = []
            continue
            
        bullet_pattern = r'^(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦►▸▹\-\*ò_~]\s*|\d+[\.\)]\s+|(?:project\s*(?:\d+|[a-z])?\s*[:–\-])|(?:title\s*[:–\-]))'
        is_bullet = bool(re.match(bullet_pattern, stripped, re.IGNORECASE))
        has_date_range = bool(re.search(r'(?:20[0-2][0-9]\s*[-to–]+\s*(?:20[0-2][0-9]|present|now|current|developing))', stripped, re.IGNORECASE))
        is_new_project = is_bullet or has_date_range
        
        if is_new_project and current_project:
            proj_text = '\n'.join(current_project)
            if len(proj_text.strip()) > 15:
                projects.append(proj_text)
            current_project = [line]
        else:
            current_project.append(line)
            
    if current_project:
        proj_text = '\n'.join(current_project)
        if len(proj_text.strip()) > 15:
            projects.append(proj_text)
            
    if len(projects) <= 1 and len(project_section_text.strip()) > 300:
        chunks = []
        curr = []
        for line in lines:
            if not line.strip(): continue
            curr.append(line)
            if sum(len(l) for l in curr) > 300:
                chunks.append('\n'.join(curr))
                curr = []
        if curr:
            chunks.append('\n'.join(curr))
        return chunks
        
    if not projects and len(project_section_text.strip()) > 15:
        projects = [project_section_text]
    return projects


def score_project_relevance(text, jd_embedding):
    """Scores project relevance against JD using BERT."""
    projects = _extract_projects(text)
    if not projects:
        return 0.0
    try:
        project_embeddings = bert_model.encode(projects, convert_to_tensor=True)
        cosine_scores = util.cos_sim(project_embeddings, jd_embedding).cpu().numpy().flatten()
        RELEVANCE_THRESHOLD = 0.3
        relevant_scores = [float(s) for s in cosine_scores if s > RELEVANCE_THRESHOLD]
        if relevant_scores:
            avg_score = sum(relevant_scores) / len(relevant_scores)
            boost = min(len(relevant_scores) * 0.05, 0.15)
            return min(avg_score + boost, 1.0)
        return 0.0
    except Exception as e:
        print(f"  Warning: project scoring error: {e}")
        return 0.0


def get_certificate_relevance(text, jd_embedding):
    """Extracts certificate mentions and scores against JD via BERT."""
    text = str(text).lower()
    cert_matches = []
    patterns = [
        r'((?:aws[\s\-]?certified|certified|certification|certificate|coursera|udemy|google[\s\-]?cloud|azure[\s\-]?certified)[^\n.,;]*)',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            cert_text = m.group(1).strip()
            if len(cert_text) > 5:
                cert_matches.append(cert_text)
    if not cert_matches:
        return 0.0
    try:
        cert_embeddings = bert_model.encode(cert_matches, convert_to_tensor=True)
        cosine_scores = util.cos_sim(cert_embeddings, jd_embedding)
        scores = cosine_scores.cpu().numpy().flatten()
        relevant_scores = [float(s) for s in scores if s > 0.3]
        if not relevant_scores:
            return 0.0
        return min(sum(relevant_scores), 1.0)
    except Exception as e:
        print(f"  Warning: cert scoring error: {e}")
        return 0.0


# ==================================================================
# STEP 3: Load Data (XLSX + JSON)
# ==================================================================
print("\n[Step 3] Loading data sources...")

# --- 3A: Load XLSX ---
df_excel = pd.DataFrame()
if os.path.exists(XLSX_PATH):
    print(f"  Loading Excel: {XLSX_PATH}")
    df_excel = pd.read_excel(XLSX_PATH)
    print(f"  Excel rows: {len(df_excel)}")

    # Build a synthetic raw_text from Excel columns for feature extraction
    skills_col = df_excel["Skills"].fillna("")
    certs_col = df_excel["Certifications"].fillna("") if "Certifications" in df_excel.columns else pd.Series([""] * len(df_excel))
    exp_col = df_excel["Experience_Years"].astype(str) + " years experience"
    email_col = df_excel["Email"].fillna("") if "Email" in df_excel.columns else pd.Series([""] * len(df_excel))
    phone_col = df_excel["Phone"].astype(str).fillna("") if "Phone" in df_excel.columns else pd.Series([""] * len(df_excel))

    df_excel["raw_text"] = (
        skills_col.astype(str) + " " +
        certs_col.astype(str) + " " +
        exp_col.astype(str) + " " +
        email_col.astype(str) + " " +
        phone_col.astype(str)
    )

    # Use the JobRole as the JD for training
    df_excel["job_description"] = df_excel["JobRole"].fillna("software engineer")
    df_excel["is_excel"] = True
else:
    print(f"  WARNING: Excel file not found at {XLSX_PATH}")

# --- 3B: Load JSON ---
df_json = pd.DataFrame()
if os.path.exists(JSON_PATH):
    print(f"  Loading JSON: {JSON_PATH}")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_rows = []
    for batch in data.get("batches", []):
        jd = batch.get("job_description", "software engineer")
        for cand in batch.get("candidates", []):
            raw_text = cand.get("raw_text", "")
            if not raw_text or not raw_text.strip():
                continue  # skip entries with no text
            json_rows.append({
                "filename": cand.get("filename", ""),
                "raw_text": raw_text,
                "job_description": jd,
                "skills_list": cand.get("matched_skills", []),
                "is_excel":    False
            })

    df_json = pd.DataFrame(json_rows)
    if not df_json.empty:
        before = len(df_json)
        df_json = df_json.drop_duplicates(subset=["filename"])
        after = len(df_json)
        print(f"  JSON resumes: {before} -> {after} (after dedup)")
else:
    print(f"  No JSON file at {JSON_PATH} (that's OK for first training)")

# --- 3C: Merge ---
df = pd.concat([df_excel, df_json], ignore_index=True)
print(f"\n  Total combined samples: {len(df)}")

if len(df) == 0:
    print("\nERROR: No training data found!")
    print(f"  Expected Excel at: {XLSX_PATH}")
    print(f"  Expected JSON at : {JSON_PATH}")
    sys.exit(1)

# ==================================================================
# STEP 4: Feature Engineering
# ==================================================================
print("\n[Step 4] Extracting features...")

# 4A: Skill count
def _get_skills_count(row):
    if row.get("is_excel", False) and "Skills" in row and pd.notna(row.get("Skills")):
        return count_skills(row["Skills"])
    elif "skills_list" in row and isinstance(row.get("skills_list"), list):
        return len(row["skills_list"])
    return count_skills(str(row.get("raw_text", "")))

df["skills_count"] = df.apply(_get_skills_count, axis=1)

# 4B: Has WORK experience (section-aware, ignores education/project timelines)
df["has_experience"] = df["raw_text"].astype(str).apply(has_work_experience)

# 4C: Has contact info
df["has_contact"] = df["raw_text"].astype(str).apply(has_contact_info)

# 4D: Experience years (section-aware)
def _get_years(row):
    if row.get("is_excel", False) and "Experience_Years" in row:
        try:
            val = row["Experience_Years"]
            if pd.notna(val):
                return min(int(float(val)), 25)
        except (ValueError, TypeError):
            pass
    return extract_experience_years(str(row.get("raw_text", "")))

df["experience_years"] = df.apply(_get_years, axis=1)

# 4E: Semantic matching via BERT (the slow step)
print("  Computing Sentence-BERT embeddings (this may take a minute)...")

jd_texts = df["job_description"].astype(str).tolist()
resume_texts = df["raw_text"].astype(str).tolist()

# Encode in batches for memory efficiency
BATCH_SIZE = 64
print(f"  Encoding {len(jd_texts)} JD texts...")
jd_embeddings = bert_model.encode(jd_texts, convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=False)
print(f"  Encoding {len(resume_texts)} resume texts...")
resume_embeddings = bert_model.encode(resume_texts, convert_to_tensor=True, batch_size=BATCH_SIZE, show_progress_bar=False)

skills_match_scores = []
cert_relevance_scores = []
project_relevance_scores = []

print("  Scoring semantic similarity + project relevance...")
for i in range(len(df)):
    # Skill match: resume text vs JD
    sim = util.cos_sim(resume_embeddings[i], jd_embeddings[i]).item()
    norm_sim = min(max((sim - 0.1) * 1.5, 0.0), 1.0)
    skills_match_scores.append(norm_sim)

    # Certificate match: individual certs vs JD
    cert_rel = get_certificate_relevance(df.iloc[i]["raw_text"], jd_embeddings[i])
    cert_relevance_scores.append(cert_rel)
    
    # Project relevance: individual projects vs JD
    proj_rel = score_project_relevance(df.iloc[i]["raw_text"], jd_embeddings[i])
    project_relevance_scores.append(proj_rel)

    if (i + 1) % 200 == 0:
        print(f"    Processed {i + 1}/{len(df)}...")

df["skills_match_score"] = skills_match_scores
df["certificate_relevance"] = cert_relevance_scores
df["project_relevance"] = project_relevance_scores
print("  Semantic features done!")

print("  Feature extraction complete!")
print(f"    skills_match_score   : mean={df['skills_match_score'].mean():.3f}, std={df['skills_match_score'].std():.3f}")
print(f"    certificate_relevance: mean={df['certificate_relevance'].mean():.3f}, std={df['certificate_relevance'].std():.3f}")
print(f"    project_relevance    : mean={df['project_relevance'].mean():.3f}, std={df['project_relevance'].std():.3f}")
print(f"    skills_count         : mean={df['skills_count'].mean():.1f}")
print(f"    experience_years     : mean={df['experience_years'].mean():.1f}")
print(f"    has_experience       : {df['has_experience'].sum()}/{len(df)}")
print(f"    has_contact          : {df['has_contact'].sum()}/{len(df)}")

# ==================================================================
# STEP 5: Generate Pseudo-Labels
# ==================================================================
print("\n[Step 5] Generating target labels via quality formula...")

# Normalize continuous vars to 0-1 range for the formula
df["exp_score"] = (df["experience_years"] / 10.0).clip(upper=1.0)
df["skills_score_norm"] = (df["skills_count"] / 15.0).clip(upper=1.0)

df["quality_score"] = (
    (df["skills_match_score"]    * SCORING_WEIGHTS["skills_match"]) +
    (df["exp_score"]             * SCORING_WEIGHTS["experience"]) +
    (df["certificate_relevance"] * SCORING_WEIGHTS["certificates"]) +
    (df["has_contact"]           * SCORING_WEIGHTS["contact_info"]) +
    (df["skills_score_norm"]     * SCORING_WEIGHTS["skills_count"]) +
    (df["project_relevance"]     * SCORING_WEIGHTS["project_relevance"])
)

df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)

strong_count = df["shortlisted"].sum()
weak_count = (df["shortlisted"] == 0).sum()

# Auto-adjust threshold if labels are too imbalanced
if strong_count < 5 or weak_count < 5:
    print(f"  Warning: Default threshold {STRONG_THRESHOLD} gave {strong_count} Strong, {weak_count} Weak.")
    print("  Auto-adjusting to top 25% as Strong...")
    STRONG_THRESHOLD = df["quality_score"].quantile(0.75)
    df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)
    strong_count = df["shortlisted"].sum()
    weak_count = (df["shortlisted"] == 0).sum()
    print(f"  New threshold: {STRONG_THRESHOLD:.3f}")

print(f"  Label distribution: {strong_count} Strong (1), {weak_count} Weak (0)")
print(f"  Quality score stats: mean={df['quality_score'].mean():.3f}, "
      f"min={df['quality_score'].min():.3f}, max={df['quality_score'].max():.3f}")

# ==================================================================
# STEP 6: Train Random Forest
# ==================================================================
print("\n[Step 6] Training Random Forest model...")

feature_cols = [
    "skills_match_score",
    "skills_count",
    "has_experience",
    "certificate_relevance",
    "has_contact",
    "experience_years",
    "project_relevance"
]
X = df[feature_cols].copy()
y = df["shortlisted"].copy()

# Fill any NaN that might have slipped through
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    class_weight="balanced",  # handles class imbalance
    random_state=42
)
model.fit(X_train, y_train)
print("  Training complete!")

print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples : {len(X_test)}")

# ==================================================================
# STEP 7: Evaluate
# ==================================================================
print("\n[Step 7] Evaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(
    y_test, y_pred,
    labels=[0, 1],
    target_names=["Weak", "Strong"],
    zero_division=0
))

# Cross-validation for more robust metric
print("  Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
print(f"  CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# --- Confusion Matrix Plot ---
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix (Threshold={STRONG_THRESHOLD:.2f})", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Weak", "Strong"])
ax.set_yticklabels(["Weak", "Strong"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
plt.tight_layout()
plt.savefig(CM_OUTPUT, dpi=150)
plt.close()
print(f"  Saved: {CM_OUTPUT}")

# --- Feature Importance Plot ---
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = [feature_cols[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
bars = ax.barh(sorted_features, sorted_importances, color=colors)
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Random Forest", fontsize=14)
for bar, val in zip(bars, sorted_importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig(FI_OUTPUT, dpi=150)
plt.close()
print(f"  Saved: {FI_OUTPUT}")

# ==================================================================
# STEP 8: Save Model
# ==================================================================
print("\n[Step 8] Saving model...")
joblib.dump(model, MODEL_OUTPUT)
print(f"  Saved: {MODEL_OUTPUT}")

# ==================================================================
# DONE
# ==================================================================
print("\n" + "=" * 60)
print("  Training Complete!")
print("=" * 60)
print(f"  Artifacts:")
print(f"    model.pkl              - Random Forest (200 trees, max_depth=10)")
print(f"    confusion_matrix.png   - Test set confusion matrix")
print(f"    feature_importance.png - Feature importance chart")
print(f"  Test Accuracy: {accuracy * 100:.2f}%")
print(f"  CV Accuracy:   {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")
print("=" * 60)
