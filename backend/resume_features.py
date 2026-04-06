# resume_features.py - Shared feature extraction & scoring config for HireFlow-AI
# =================================================================================
# This module is the SINGLE SOURCE OF TRUTH for:
#   - Skill keyword vocabulary (300+ cross-industry terms)
#   - Scoring weights configuration
#   - Section splitting (resume → work, education, projects, skills, etc.)
#   - Context detection (education vs work vs project)
#   - Project extraction & splitting
#   - Experience extraction (years, has_experience)
#   - Contact info detection
#   - Skill counting & matching
#   - Education quality scoring
#   - Name extraction from filenames
#
# Both candidate_scorer.py and train_model.py import from this module,
# eliminating ~400 lines of duplication.
# =================================================================================

import os
import re
import datetime

# ====================================================================
# SCORING CONFIGURATION
# Adjust these weights based on what matters most for shortlisting.
# Must sum to 1.0.
# ====================================================================
SCORING_WEIGHTS = {
    "jd_skill_overlap":   0.22,   # Direct keyword match
    "skills_match":       0.15,   # BERT semantic similarity
    "experience":         0.13,   # Years of WORK experience only
    "certificates":       0.10,   # Relevant certificates
    "contact_info":       0.05,   # Has email, phone, linkedin
    "skills_count":       0.10,   # Raw count of distinct skills
    "project_relevance":  0.15,   # Relevant projects matching JD
    "education_quality":  0.10,   # Degree level + field relevance
}
STRONG_THRESHOLD = 0.5  # Score >= 50% is considered a "Strong" candidate (Class 1)

# Normalisation cap for experience years
EXPERIENCE_NORM_CAP = 15.0

# Experience floor: fresh grads with 0 detected years still get this baseline
EXPERIENCE_FLOOR = 0.1

# ====================================================================
# SKILL KEYWORD VOCABULARY (300+ cross-industry terms)
# ====================================================================
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
# Section Splitting - Parses resume into logical sections
# ============================================================

# Patterns that identify section headings in resumes
# Supports both standalone headings (e.g., "SKILLS") and
# inline headings with content (e.g., "Skills: Python, Java")
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

# Inline section heading pattern: "HEADING: content on same line"
# E.g., "Skills: Python, Java, AWS" or "Experience: 5 years in..."
_INLINE_SECTION_PATTERNS = {
    "work": re.compile(
        r'^\s*(?:work\s*experience|professional\s*experience|employment\s*history|'
        r'employment|work\s*history)\s*:\s*\S',
        re.IGNORECASE
    ),
    "education": re.compile(
        r'^\s*(?:education|academic\s*background|qualifications?)\s*:\s*\S',
        re.IGNORECASE
    ),
    "projects": re.compile(
        r'^\s*(?:projects?|academic\s*projects?|personal\s*projects?|'
        r'key\s*projects?|major\s*projects?)\s*:\s*\S',
        re.IGNORECASE
    ),
    "skills": re.compile(
        r'^\s*(?:skills?|technical\s*skills?|core\s*competenc(?:ies|e)|'
        r'key\s*skills?)\s*:\s*\S',
        re.IGNORECASE
    ),
    "certificates": re.compile(
        r'^\s*(?:certifications?|certificates?|professional\s*certifications?)\s*:\s*\S',
        re.IGNORECASE
    ),
    "summary": re.compile(
        r'^\s*(?:summary|objective|profile|career\s*objective|professional\s*summary)\s*:\s*\S',
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

# Education degree patterns for quality scoring
_DEGREE_PATTERNS = {
    "phd": re.compile(r'\b(?:ph\.?d|doctorate|doctor\s*of\s*philosophy)\b', re.IGNORECASE),
    "masters": re.compile(
        r'\b(?:m\.?tech|m\.?sc|m\.?s\b|m\.?e\b|m\.?a\b|m\.?com|mba|mca|'
        r'master(?:\'?s)?(?:\s*of|\s*in|\s*degree)?)\b',
        re.IGNORECASE
    ),
    "bachelors": re.compile(
        r'\b(?:b\.?tech|b\.?sc|b\.?s\b|b\.?e\b|b\.?a\b|b\.?com|bba|bca|'
        r'bachelor(?:\'?s)?(?:\s*of|\s*in|\s*degree)?)\b',
        re.IGNORECASE
    ),
    "diploma": re.compile(
        r'\b(?:diploma|associate(?:\'?s)?\s*degree|certificate\s*program)\b',
        re.IGNORECASE
    ),
}

# Degree level scores
_DEGREE_SCORES = {
    "phd": 1.0,
    "masters": 0.8,
    "bachelors": 0.6,
    "diploma": 0.4,
}

# JD-related term expansion for common role descriptions
_JD_RELATED_TERMS = {
    "web development": ["html", "css", "javascript", "react", "angular", "vue.js", "node.js", "django", "flask"],
    "full stack": ["html", "css", "javascript", "react", "node.js", "sql", "mongodb", "postgresql", "mysql"],
    "frontend": ["html", "css", "javascript", "react", "angular", "vue.js", "typescript"],
    "backend": ["node.js", "python", "java", "sql", "mongodb", "postgresql", "flask", "django", "spring boot"],
    "devops": ["docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible", "aws", "azure", "gcp"],
    "data engineer": ["sql", "python", "pandas", "data analysis", "data science", "tensorflow"],
    "mobile development": ["react", "javascript", "typescript", "flutter", "swift"],
}


# ============================================================
# Section Splitting
# ============================================================

def split_resume_sections(text):
    """
    Splits resume raw text into named sections based on common headings.
    Returns a dict like:
        {"work": "...", "education": "...", "projects": "...", "other": "..."}
    Supports both standalone headings and inline "HEADING: content" patterns.
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

        # Check if this line is a standalone section heading
        matched_section = None
        for section_name, pattern in _SECTION_PATTERNS.items():
            if pattern.match(stripped):
                matched_section = section_name
                section_found = True
                break

        if matched_section:
            current_section = matched_section
        else:
            # Check for inline headings like "Skills: Python, Java"
            inline_matched = None
            for section_name, pattern in _INLINE_SECTION_PATTERNS.items():
                if pattern.match(stripped):
                    inline_matched = section_name
                    section_found = True
                    break

            if inline_matched:
                current_section = inline_matched
                # Extract the content after the colon and add it to the section
                colon_pos = stripped.find(':')
                if colon_pos >= 0:
                    content_after = stripped[colon_pos + 1:].strip()
                    if content_after:
                        sections[current_section].append(content_after)
            else:
                sections[current_section].append(line)

    # Convert lists to strings
    result = {k: '\n'.join(v) for k, v in sections.items()}
    result["_sections_found"] = section_found

    return result


# ============================================================
# Context Detection Helpers
# ============================================================

def is_education_context(surrounding_text):
    """Check if the surrounding text (a few lines around a date range) is education-related."""
    return bool(_EDUCATION_CONTEXT_KEYWORDS.search(surrounding_text))


def is_project_context(surrounding_text):
    """Check if the surrounding text is project-related."""
    return bool(_PROJECT_CONTEXT_KEYWORDS.search(surrounding_text))


def is_work_context(surrounding_text):
    """Check if the surrounding text is work/employment-related."""
    return bool(_WORK_CONTEXT_KEYWORDS.search(surrounding_text))


# ============================================================
# Feature Extraction
# ============================================================

def count_skills(matched_skills):
    """Returns the raw count of matched skills."""
    if isinstance(matched_skills, list):
        return len(matched_skills)
    # Handle pandas NaN or raw text
    import pandas as pd
    if pd.isna(matched_skills):
        return 0
    text = str(matched_skills).lower()
    count = 0
    for s in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(s) + r'\b'
        if re.search(pattern, text):
            count += 1
    return count


def has_contact_info(text):
    """Returns 1 if the text contains an email, phone number, or LinkedIn URL."""
    text = str(text).lower()
    has_email    = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone    = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0


def has_work_experience(text):
    """
    Returns 1 if ACTUAL work experience is found (not education or project timelines).
    Uses section-aware parsing: only looks for experience indicators in the work section.
    Falls back to context-based detection if sections can't be parsed.
    """
    text = str(text)
    sections = split_resume_sections(text)

    work_text = sections.get("work", "").lower()

    if sections["_sections_found"] and work_text.strip():
        # Section headers were found and we have work section content
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
            if is_education_context(context) or is_project_context(context):
                continue
            if is_work_context(context):
                return 1
            # If no clear context, still check if it's NOT education/project
            if not is_education_context(context) and not is_project_context(context):
                if re.search(r'(pvt|ltd|inc|corp|llc|company|firm|technologies|solutions|services)', context):
                    return 1

    # Also check for explicit "X years of experience" pattern anywhere
    if re.search(r'\d+\s*\+?\s*years?\s+of\s+(?:work\s+)?experience', text_lower):
        return 1
    if re.search(r'(worked at|employed at|working at|employment history)', text_lower):
        return 1

    return 0


def extract_experience_years(text):
    """
    Extracts years of WORK experience only.
    Ignores education timelines (degree durations) and project timelines.
    Uses section-aware parsing when possible, falls back to context detection.
    """
    text = str(text)
    current_year = datetime.datetime.now().year
    sections = split_resume_sections(text)

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
        if is_education_context(context):
            continue
        if is_project_context(context):
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
# Project Extraction & Splitting
# ============================================================

def extract_projects(text):
    """
    Extracts individual project descriptions from the resume.
    Returns a list of project text strings.
    """
    text = str(text)
    sections = split_resume_sections(text)

    project_text = sections.get("projects", "")

    if sections["_sections_found"] and project_text.strip():
        # We have a clear projects section — split it into individual projects
        return split_individual_projects(project_text)

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


def split_individual_projects(project_section_text):
    """
    Splits a project section into individual projects.
    Uses bullet points, numbered lists, date ranges, title-case heuristics,
    or fallback chunking.
    """
    lines = project_section_text.split('\n')
    projects = []
    current_project = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current_project:
                proj_text = '\n'.join(current_project)
                if len(proj_text.strip()) > 15:
                    projects.append(proj_text)
                current_project = []
            continue

        # Extended bullet characters incl. corrupted ones like ò, and _
        bullet_pattern = r'^(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦►▸▹\-\*ò_~]\s*|\d+[\.\\)]\s+|(?:project\s*(?:\d+|[a-z])?\s*[:–\-])|(?:title\s*[:–\-]))'
        is_bullet = bool(re.match(bullet_pattern, stripped, re.IGNORECASE))

        # Does it contain a date range on this line? (common for project headers)
        has_date_range = bool(re.search(r'(?:20[0-2][0-9]\s*[-to–]+\s*(?:20[0-2][0-9]|present|now|current|developing))', stripped, re.IGNORECASE))

        # Title-case heuristic: short line, mostly title-cased words, after a blank line
        # E.g., "E-commerce Website" or "AI Chatbot System"
        is_title_case = False
        if len(stripped) < 80 and not is_bullet and not has_date_range:
            words = stripped.split()
            # At least 2 words, most of them capitalized (ignoring small words)
            small_words = {'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', '&', '-', '–'}
            if len(words) >= 2:
                cap_words = [w for w in words if w.lower() in small_words or w[0].isupper()]
                if len(cap_words) >= len(words) * 0.7:
                    # Check that the previous line was blank or this is the first line
                    prev_blank = (idx == 0) or (not lines[idx - 1].strip())
                    if prev_blank:
                        is_title_case = True

        is_new_project = is_bullet or has_date_range or is_title_case

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

    # Fallback chunking if we STILL only have 1 giant project
    if len(projects) <= 1 and len(project_section_text.strip()) > 300:
        chunks = []
        curr = []
        for line in lines:
            if not line.strip():
                continue
            curr.append(line)
            # ~300 chars per chunk to avoid Sentence-BERT truncation
            if sum(len(l) for l in curr) > 300:
                chunks.append('\n'.join(curr))
                curr = []
        if curr:
            chunks.append('\n'.join(curr))
        return chunks

    if not projects and len(project_section_text.strip()) > 15:
        projects = [project_section_text]

    return projects


# ============================================================
# Skill Matching
# ============================================================

def get_matched_skills(raw_text):
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


def get_jd_skills(job_description):
    """Extracts which SKILL_KEYWORDS are mentioned in the JD itself.
    Also extracts related terms from role descriptions and simple n-grams."""
    if not job_description or not job_description.strip():
        return []

    jd_lower = job_description.lower()
    jd_skills = []

    # 1. Match against the vocabulary
    for s in SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(s) + r'\b'
        if re.search(pattern, jd_lower):
            jd_skills.append(s)

    # 2. Expand related terms via role description mapping
    for phrase, related_skills in _JD_RELATED_TERMS.items():
        if phrase in jd_lower:
            for skill in related_skills:
                if skill not in jd_skills:
                    jd_skills.append(skill)

    # 3. Extract n-grams from JD that look like skill/tool names
    #    (2-3 word phrases that are title-cased or contain technical patterns)
    jd_words = re.findall(r'[A-Za-z][A-Za-z0-9\+\#\.\-/]{1,}', job_description)
    for i in range(len(jd_words)):
        # Bigrams
        if i + 1 < len(jd_words):
            bigram = f"{jd_words[i]} {jd_words[i+1]}".lower()
            if bigram not in jd_skills and len(bigram) > 5:
                # Only add if it looks technical (not common English)
                common_words = {'the', 'and', 'for', 'with', 'has', 'have', 'are', 'was', 'will', 'can',
                               'should', 'must', 'our', 'their', 'this', 'that', 'from', 'into', 'also',
                               'need', 'looking', 'seeking', 'required', 'preferred', 'experience', 'years',
                               'strong', 'good', 'excellent', 'ability', 'team', 'work', 'working'}
                if jd_words[i].lower() not in common_words and jd_words[i+1].lower() not in common_words:
                    # Check if it matches something in any resume (skip for now, just add)
                    pass  # n-gram expansion is handled by BERT similarity instead

    return jd_skills


def compute_jd_overlap(matched_skills, jd_skills):
    """Computes what fraction of JD-required skills are found in the resume.
    Returns a score from 0.0 to 1.0."""
    if not jd_skills:
        return 0.0

    matched_set = set(matched_skills)
    jd_set = set(jd_skills)

    overlap = matched_set & jd_set

    # Score: fraction of JD skills found in resume
    return len(overlap) / len(jd_set)


# ============================================================
# Education Quality Scoring
# ============================================================

def extract_education_quality(text, job_description=""):
    """
    Scores the candidate's education quality from 0.0 to 1.0.
    Considers:
    - Degree level (PhD=1.0, Masters=0.8, Bachelors=0.6, Diploma=0.4, None=0.0)
    - Field relevance to the JD (bonus if the degree field matches JD keywords)

    Returns a float score between 0.0 and 1.0.
    """
    text = str(text)
    sections = split_resume_sections(text)

    # Prefer the education section, but fall back to full text
    edu_text = sections.get("education", "")
    if not edu_text.strip():
        edu_text = text

    edu_lower = edu_text.lower()

    # Find the highest degree level
    degree_score = 0.0
    for degree_name, pattern in _DEGREE_PATTERNS.items():
        if pattern.search(edu_lower):
            candidate_score = _DEGREE_SCORES[degree_name]
            if candidate_score > degree_score:
                degree_score = candidate_score

    # Field relevance bonus: check if the education text mentions JD-related terms
    field_bonus = 0.0
    if job_description and job_description.strip():
        jd_lower = job_description.lower()
        jd_words = set(re.findall(r'\b[a-z]{3,}\b', jd_lower))
        edu_words = set(re.findall(r'\b[a-z]{3,}\b', edu_lower))

        # Remove very common words
        stop_words = {'the', 'and', 'for', 'with', 'has', 'have', 'are', 'was', 'will', 'can',
                      'from', 'this', 'that', 'not', 'but', 'also', 'any', 'all', 'been', 'more',
                      'years', 'year', 'experience', 'required', 'preferred'}
        jd_words -= stop_words
        edu_words -= stop_words

        if jd_words:
            overlap = jd_words & edu_words
            if len(overlap) >= 2:
                field_bonus = 0.2  # Meaningful field overlap
            elif len(overlap) >= 1:
                field_bonus = 0.1

    return min(degree_score + field_bonus, 1.0)


# ============================================================
# Certificate Extraction (Broadened Patterns)
# ============================================================

def extract_certificate_mentions(text):
    """
    Extracts certificate/certification mentions from resume text.
    Uses both regex patterns and section-based extraction.
    Returns a list of certificate text strings.
    """
    text_lower = str(text).lower()
    sections = split_resume_sections(text)
    cert_matches = []

    # Pattern-based extraction from full text
    cert_patterns = [
        r'((?:aws[\s\-]?certified|certified|certification|certificate|'
        r'coursera|udemy|google[\s\-]?cloud|azure[\s\-]?certified|'
        r'professional\s*certificate|nanodegree|specialization|'
        r'licensed|registered)[^\n.,;]*)',
    ]
    for pattern in cert_patterns:
        for m in re.finditer(pattern, text_lower):
            cert_text = m.group(1).strip()
            if len(cert_text) > 5:
                cert_matches.append(cert_text)

    # Section-based extraction: if we have a certificates section, extract lines from it
    cert_section = sections.get("certificates", "").strip()
    if cert_section:
        for line in cert_section.split('\n'):
            line = line.strip()
            if len(line) > 5 and line.lower() not in [c.lower() for c in cert_matches]:
                cert_matches.append(line)

    return cert_matches


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
