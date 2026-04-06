# candidate_scorer.py - scores and ranks candidates using Random Forest ML Model + NLP Semantic Matching
# ============================================================================================
# This module handles the scoring of individual resumes against a Job Description.
# PATH A: If model.pkl exists -> uses Random Forest + Sentence-BERT (full ML pipeline)
# PATH B: If model.pkl is missing -> falls back to simple keyword matching
# ============================================================================================

import os
import re
import statistics
import datetime
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ====================================================================
# SCORING CONFIGURATION (EDITABLE!)
# Adjust these weights based on what matters most for shortlisting.
# Must sum to 1.0.
# ====================================================================
SCORING_WEIGHTS = {
    "jd_skill_overlap":   0.25,   # Direct keyword match
    "skills_match":       0.15,   # BERT semantic similarity
    "experience":         0.15,   # Years of WORK experience only
    "certificates":       0.10,   # Relevant certificates
    "contact_info":       0.10,   # Has email, phone, linkedin
    "skills_count":       0.10,   # Raw count of distinct skills
    "project_relevance":  0.15,   # Relevant projects matching JD
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

_model      = None
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
        from sentence_transformers import SentenceTransformer
        _bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[candidate_scorer] Sentence-BERT ready!")


# ============================================================
# Section Splitting - Parses resume into logical sections
# ============================================================

# Patterns that identify section headings in resumes
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

# Keywords that indicate education context (used as fallback when sections can't be parsed)
_EDUCATION_CONTEXT_KEYWORDS = re.compile(
    r'(?:b\.?tech|m\.?tech|b\.?sc|m\.?sc|b\.?e\b|m\.?e\b|b\.?a\b|m\.?a\b|b\.?com|m\.?com|'
    r'bachelor|master|ph\.?d|diploma|degree|university|college|institute|'
    r'school|gpa|cgpa|percentage|semester|graduated|graduation|'
    r'higher\s*secondary|hsc|ssc|10th|12th|board|cbse|icse|'
    r'mba|bba|bca|mca|enrolled)',
    re.IGNORECASE
)

# Keywords that indicate project context
_PROJECT_CONTEXT_KEYWORDS = re.compile(
    r'(?:project|capstone|mini[\s\-]?project|course\s*work|coursework|'
    r'hackathon|competition|challenge|assignment|thesis|dissertation|'
    r'research\s*paper|paper\s*titled|paper\s*on)',
    re.IGNORECASE
)

# Keywords that indicate actual work/employment context  
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
    """
    Splits resume raw text into named sections based on common headings.
    Returns a dict like:
        {"work": "...", "education": "...", "projects": "...", "other": "..."}
    If no clear sections found, uses keyword-based context detection as fallback.
    """
    text = str(text)
    lines = text.split('\n')
    
    sections = {
        "work": [],
        "education": [],
        "projects": [],
        "skills": [],
        "certificates": [],
        "summary": [],
        "other": [],
    }
    
    current_section = "other"
    section_found = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            sections[current_section].append(line)
            continue
        
        # Check if this line is a section heading
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
    
    # Convert lists to strings
    result = {k: '\n'.join(v) for k, v in sections.items()}
    result["_sections_found"] = section_found
    
    return result


def _is_education_context(surrounding_text):
    """Check if the surrounding text (a few lines around a date range) is education-related."""
    return bool(_EDUCATION_CONTEXT_KEYWORDS.search(surrounding_text))


def _is_project_context(surrounding_text):
    """Check if the surrounding text is project-related."""
    return bool(_PROJECT_CONTEXT_KEYWORDS.search(surrounding_text))


def _is_work_context(surrounding_text):
    """Check if the surrounding text is work/employment-related."""
    return bool(_WORK_CONTEXT_KEYWORDS.search(surrounding_text))


# ============================================================
# Feature Extraction Helpers 
# ============================================================

def _count_skills(matched_skills):
    """Returns the raw count of matched skills."""
    return len(matched_skills)


def _has_contact_info(text):
    """Returns 1 if the text contains an email, phone number, or LinkedIn URL."""
    text = str(text).lower()
    has_email    = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone    = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0


def _has_work_experience(text):
    """
    Returns 1 if ACTUAL work experience is found (not education or project timelines).
    Uses section-aware parsing: only looks for experience indicators in the work section.
    Falls back to context-based detection if sections can't be parsed.
    """
    text = str(text)
    sections = _split_resume_sections(text)
    
    work_text = sections.get("work", "").lower()
    
    if sections["_sections_found"] and work_text.strip():
        # Section headers were found and we have work section content
        # Check for work-related keywords or date ranges in the work section
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
    
    # Fallback: no clear section headers found — use full text but with context filtering
    text_lower = text.lower()
    lines = text_lower.split('\n')
    
    for i, line in enumerate(lines):
        # Look for date ranges in this line
        if re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now|current))', line):
            # Get surrounding context (3 lines before and after)
            context_start = max(0, i - 3)
            context_end = min(len(lines), i + 4)
            context = ' '.join(lines[context_start:context_end])
            
            # Only count as experience if it's work context, not education or project
            if _is_education_context(context) or _is_project_context(context):
                continue
            if _is_work_context(context):
                return 1
            # If no clear context, still check if it's NOT education/project
            if not _is_education_context(context) and not _is_project_context(context):
                # Check if there are company-like indicators nearby
                if re.search(r'(pvt|ltd|inc|corp|llc|company|firm|technologies|solutions|services)', context):
                    return 1
    
    # Also check for explicit "X years of experience" pattern anywhere
    if re.search(r'\d+\s*\+?\s*years?\s+of\s+(?:work\s+)?experience', text_lower):
        return 1
    if re.search(r'(worked at|employed at|working at|employment history)', text_lower):
        return 1
    
    return 0


def _extract_experience_years(text):
    """
    Extracts years of WORK experience only.
    Ignores education timelines (degree durations) and project timelines.
    Uses section-aware parsing when possible, falls back to context detection.
    """
    text = str(text)
    current_year = datetime.datetime.now().year
    sections = _split_resume_sections(text)
    
    work_text = sections.get("work", "")
    
    # Strategy 1: If we have a clear work section, use only that
    if sections["_sections_found"] and work_text.strip():
        work_lower = work_text.lower()
        
        # Look for explicit "X years of experience"
        match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(work\s+)?(experience)?', work_lower)
        if match:
            return min(int(match.group(1)), 25)
        
        # Sum up date ranges in the work section only
        total_years = 0
        date_ranges = re.finditer(
            r'(20[0-2][0-9])\s*[-to–]+\s*(20[0-2][0-9]|present|now|current)',
            work_lower
        )
        for dr in date_ranges:
            start_year = int(dr.group(1))
            end_str = dr.group(2)
            end_year = current_year if end_str in ['present', 'now', 'current'] else int(end_str)
            if end_year >= start_year:
                total_years += (end_year - start_year)
        
        if total_years > 0:
            return min(total_years, 25)
        return 0
    
    # Strategy 2: Fallback — context-based filtering on full text
    text_lower = text.lower()
    
    # First check for explicit "X years of experience" (high confidence)
    match = re.search(r'(\d+)\s*\+?\s*years?\s+of\s+(?:work\s+)?experience', text_lower)
    if match:
        return min(int(match.group(1)), 25)
    
    # Now process date ranges with context awareness
    lines = text_lower.split('\n')
    total_years = 0
    
    for i, line in enumerate(lines):
        date_ranges = list(re.finditer(
            r'(20[0-2][0-9])\s*[-to–]+\s*(20[0-2][0-9]|present|now|current)',
            line
        ))
        
        if not date_ranges:
            continue
        
        # Get surrounding context
        context_start = max(0, i - 3)
        context_end = min(len(lines), i + 4)
        context = ' '.join(lines[context_start:context_end])
        
        # Skip education and project timelines
        if _is_education_context(context):
            continue
        if _is_project_context(context):
            continue
        
        # Count this date range as work experience
        for dr in date_ranges:
            start_year = int(dr.group(1))
            end_str = dr.group(2)
            end_year = current_year if end_str in ['present', 'now', 'current'] else int(end_str)
            if end_year >= start_year:
                total_years += (end_year - start_year)
    
    if total_years > 0:
        return min(total_years, 25)
    
    return 0


# ============================================================
# Project Extraction & Relevance Scoring
# ============================================================

def _extract_projects(text):
    """
    Extracts individual project descriptions from the resume.
    Returns a list of project text strings.
    """
    text = str(text)
    sections = _split_resume_sections(text)
    
    project_text = sections.get("projects", "")
    
    if sections["_sections_found"] and project_text.strip():
        # We have a clear projects section — split it into individual projects
        return _split_individual_projects(project_text)
    
    # Fallback: look for project-like blocks in the full text
    projects = []
    lines = text.split('\n')
    
    in_project = False
    current_project = []
    
    for line in lines:
        stripped = line.strip().lower()
        
        # Detect project headings like "Project: XYZ" or "Project Title: XYZ"
        if re.match(r'(?:project\s*(?:title|name)?\s*[:–\-])', stripped):
            if current_project:
                projects.append('\n'.join(current_project))
            current_project = [line]
            in_project = True
        elif in_project:
            # Check if we've hit a new section heading (exit project block)
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
    """
    Splits a project section into individual projects.
    Uses bullet points, numbered lists, or blank-line separation as delimiters.
    """
    lines = project_section_text.split('\n')
    projects = []
    current_project = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_project:
                proj_text = '\n'.join(current_project)
                if len(proj_text.strip()) > 15:  # Skip noise
                    projects.append(proj_text)
                current_project = []
            continue
        
        # Project delimiter: bullet point, numbered list, or title-like pattern
        is_new_project = bool(re.match(
            r'^(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦►▸▹\-\*]\s+|'  # bullets
            r'\d+[\.\)]\s+|'  # numbered
            r'(?:project\s*(?:\d+|[a-z])?\s*[:–\-])|'  # "Project 1:" etc.
            r'(?:title\s*[:–\-]))',  # "Title:"
            stripped, re.IGNORECASE
        ))
        
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
    
    # If no individual projects were split, treat the entire section as one project
    if not projects and len(project_section_text.strip()) > 15:
        projects = [project_section_text]
    
    return projects


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
    
    projects = _extract_projects(raw_text)
    
    if not projects:
        return 0.0, 0
    
    if not job_description or not job_description.strip():
        job_description = "software engineer developer"
    
    try:
        jd_embedding = _bert_model.encode([job_description], convert_to_tensor=True)[0]
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


def _get_matched_skills(raw_text):
    """Finds which baseline skills are explicitly in the text (for dashboard display).
    Uses word-boundary regex to prevent false positives like 'go' matching 'google'."""
    text_lower = raw_text.lower()
    matched = []
    for s in SKILL_KEYWORDS:
        # Build a word-boundary pattern. re.escape handles special chars like c++, c#, etc.
        pattern = r'\b' + re.escape(s) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(s)
    return matched


def _get_jd_skills(job_description):
    """Extracts which SKILL_KEYWORDS are mentioned in the JD itself.
    Also extracts simple multi-word phrases like 'web development', 'full stack'."""
    if not job_description or not job_description.strip():
        return []
    
    jd_lower = job_description.lower()
    jd_skills = []
    for s in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(s) + r'\b'
        if re.search(pattern, jd_lower):
            jd_skills.append(s)
    
    # Also check for common related terms that aren't in the keyword list
    extra_terms = {
        "web development": ["html", "css", "javascript", "react", "angular", "vue.js", "node.js", "django", "flask"],
        "full stack": ["html", "css", "javascript", "react", "node.js", "sql", "mongodb", "postgresql", "mysql"],
        "frontend": ["html", "css", "javascript", "react", "angular", "vue.js", "typescript"],
        "backend": ["node.js", "python", "java", "sql", "mongodb", "postgresql", "flask", "django", "spring boot"],
        "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible", "aws", "azure", "gcp"],
        "data engineer": ["sql", "python", "pandas", "data analysis", "data science", "tensorflow"],
        "mobile development": ["react", "javascript", "typescript", "flutter", "swift"],
    }
    
    for phrase, related_skills in extra_terms.items():
        if phrase in jd_lower:
            for skill in related_skills:
                if skill not in jd_skills:
                    jd_skills.append(skill)
    
    return jd_skills


def _compute_jd_overlap(matched_skills, jd_skills):
    """Computes what fraction of JD-required skills are found in the resume.
    Returns a score from 0.0 to 1.0."""
    if not jd_skills:
        return 0.0
    
    matched_set = set(matched_skills)
    jd_set = set(jd_skills)
    
    overlap = matched_set & jd_set
    
    # Score: fraction of JD skills found in resume
    return len(overlap) / len(jd_set)


def _get_semantic_features(raw_text, job_description):
    """
    Uses Sentence-BERT to compute:
    1. Overall skill/resume meaning vs Job Description  → skills_match_score (0-1)
    2. Certificate relevance vs Job Description         → cert_relevance_score (0-1)
    """
    from sentence_transformers import util

    if not job_description or not job_description.strip():
        job_description = "software engineer developer"


    raw_text = str(raw_text).lower()

    # Pre-embed JD
    jd_embedding = _bert_model.encode([job_description], convert_to_tensor=True)[0]

    # 1. Overall Skills Semantic Score
    resume_embedding = _bert_model.encode([raw_text], convert_to_tensor=True)[0]

    # Overall semantic similarity (normalized)
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
    matched_skills = _get_matched_skills(raw_text)
    
    # Compute JD-specific skill overlap
    jd_skills = _get_jd_skills(job_description)
    jd_overlap = _compute_jd_overlap(matched_skills, jd_skills)
    
    # Find which of the candidate's skills directly match the JD (for highlighting)
    jd_matched_skills = [s for s in matched_skills if s in set(jd_skills)]

    # Initialize variables for return values so they exist even if ML fails
    exp_years = _extract_experience_years(raw_text)
    cert_score = 0.0
    project_score = 0.0
    relevant_projects = 0

    # -------------------------------------------------------
    # PATH A: ML pipeline is loaded
    # -------------------------------------------------------
    if _model is not None and _bert_model is not None:
        skills_count = _count_skills(matched_skills)
        has_exp = _has_work_experience(raw_text)
        has_contact = _has_contact_info(raw_text)
        exp_years = _extract_experience_years(raw_text)

        # NLP Semantic scores
        sm_score, cert_score = _get_semantic_features(raw_text, job_description)
        
        # Project relevance scoring
        project_score, relevant_projects = _score_project_relevance(raw_text, job_description)

        # Features ordered EXACTLY as trained in train_model.py
        feature_cols = [
            "skills_match_score",
            "skills_count",
            "has_experience",
            "certificate_relevance",
            "has_contact",
            "experience_years",
            "project_relevance"
        ]

        X_infer = pd.DataFrame(
            [[sm_score, skills_count, has_exp, cert_score, has_contact, exp_years, project_score]],
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
        
        # Calculate deterministic quality score with JD overlap as the primary signal
        quality_score = (
            (jd_overlap * SCORING_WEIGHTS["jd_skill_overlap"]) +
            (sm_score * SCORING_WEIGHTS["skills_match"]) +
            (min(exp_years / 15.0, 1.0) * SCORING_WEIGHTS["experience"]) +
            (cert_score * SCORING_WEIGHTS["certificates"]) +
            (has_contact * SCORING_WEIGHTS["contact_info"]) +
            (min(skills_count / 15.0, 1.0) * SCORING_WEIGHTS["skills_count"]) +
            (project_score * SCORING_WEIGHTS["project_relevance"])
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
