# HireFlow-AI: ML Model Training & Directory Cleanup (v2)

## Background

Rebuild the ML pipeline with a proper Random Forest model that uses semantic understanding, configurable scoring weights, and smart certificate/skill matching against the Job Description.

---

## Decisions Based on Your Feedback

### 1. Data Flow
- **XLSX** = initial training data (1000 structured rows to bootstrap the model)
- **JSON** = the live data store. Every uploaded resume is extracted → scored → saved to JSON. In the final deployment, the model can be retrained periodically from this ever-growing JSON file.
- Both sources are used for the first training run, then JSON becomes the primary source going forward.

### 2. Certificate Scoring (Smart, Not Binary)
- NOT a simple "has certificate = +1". Instead:
- Use **Sentence-BERT** to compare each detected certificate against the job description
- If the JD says "AI/ML Engineer" and the resume has "AWS Cloud Practitioner" → low relevance score
- If the JD says "AI/ML Engineer" and the resume has "TensorFlow Developer Certificate" → high relevance score
- **Multiple relevant certificates = higher score** (capped at a reasonable max)

### 3. Editable Weights
- All feature weights are stored in a **config dictionary** that can be changed anytime:
```python
SCORING_WEIGHTS = {
    "skills_match":       0.35,   # semantic skill matching vs JD
    "experience":         0.25,   # years of experience
    "certificates":       0.20,   # relevant certificates (semantic match vs JD)
    "contact_info":       0.10,   # has email, phone, linkedin
    "skills_count":       0.10,   # total number of distinct skills
}
```
- A company that values experience more can simply change `"experience": 0.40` and reduce others.
- This config lives in `candidate_scorer.py` and is easy to expose via an API endpoint later.

### 4. Strong/Weak Threshold
- **Strong = quality_score >= 0.6** (not 0.45 as I originally proposed)
- This threshold is also **editable** in the config.

### 5. Database Strategy
- **Remove `database.py` and SQLite for now** — it's redundant with JSON storage.
- **Future recommendation: PostgreSQL on AWS RDS** because:
  - It supports JSON/JSONB columns natively (so you can store raw resume text + structured fields in one table)
  - It scales well on AWS RDS (managed backups, read replicas)
  - It has strong Python support via `psycopg2` or `SQLAlchemy`
  - It's free-tier eligible on AWS RDS
- When the time comes, we'll create a new `database.py` with PostgreSQL + SQLAlchemy ORM.

### 6. mock_s3.py → Remove
- It's just a `print()` statement — not useful as a placeholder.
- When you integrate real S3, you'll use `boto3` directly. We'll build that fresh when needed.

### 7. generate_100_resumes.py → Keep
- It generates contextual synthetic resumes with real skill combinations and anomaly simulation. Useful for testing.

---

## Files to Delete

| File | Reason |
|------|--------|
| `backend/model.pkl` | Old model, will be retrained |
| `backend/label_encoders.pkl` | Old encoders, not needed in new pipeline |
| `backend/confusion_matrix.png` | Old plot, will be regenerated |
| `backend/feature_importance.png` | Old plot, will be regenerated |
| `backend/__pycache__/` | Python cache, auto-regenerates |
| `backend/test_extraction.py` | One-off test script |
| `backend/test_upload.py` | One-off test script |
| `backend/mock_s3.py` | Just a print statement, not useful |
| `backend/database.py` | Removing DB layer for now (PostgreSQL later) |
| `backend/hireflow_dev.db` | SQLite database file |
| Contents of `backend/extracted/` | Leftover files from past uploads |
| Contents of `backend/uploads/` | Leftover uploaded ZIPs |

---

## Proposed Changes

### Component 1: Directory Cleanup
Delete all files listed above. Remove `mock_s3` and `database` imports from `app.py`.

---

### Component 2: Rewritten Training Script

#### [MODIFY] [train_model.py](file:///e:/hireflow-ai/backend/train_model.py)

Complete rewrite with the following pipeline:

**Step 1 — Load & Merge Data:**
- Load XLSX (1000 structured rows with Skills, Experience_Years, Certifications, etc.)
- Load JSON, deduplicate by `filename`, extract features from `raw_text`
- Merge into a single training DataFrame

**Step 2 — Feature Engineering (6 features):**

| Feature | Type | Extraction Method |
|---------|------|-------------------|
| `skills_match_score` | Float (0–1) | Sentence-BERT cosine similarity: candidate skills vs a reference JD/skills embedding |
| `skills_count` | Integer | Count of distinct skills detected in resume text |
| `has_experience` | Binary (0/1) | Regex for "X years", date ranges, keywords like "worked at", "experience" |
| `certificate_relevance` | Float (0–1) | Sentence-BERT similarity of each detected certificate vs the JD. Multiple relevant certs = higher score (capped). Irrelevant certs contribute nothing. |
| `has_contact` | Binary (0/1) | Regex for email, phone numbers, LinkedIn URLs |
| `experience_years` | Integer | Parsed numeric years from text |

**Step 3 — Pseudo-Label Generation:**
```python
quality_score = (skills_match_score    * SCORING_WEIGHTS["skills_match"]) +
                (experience_score      * SCORING_WEIGHTS["experience"]) +
                (certificate_relevance * SCORING_WEIGHTS["certificates"]) +
                (has_contact           * SCORING_WEIGHTS["contact_info"]) +
                (skills_count_norm     * SCORING_WEIGHTS["skills_count"])

label = 1 if quality_score >= 0.6 else 0   # "Strong" vs "Weak" — EDITABLE
```

**Step 4 — Train Random Forest:**
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

**Step 5 — Evaluate & Save:**
- Print accuracy + classification report
- Save `model.pkl` (Random Forest) and `feature_config.pkl` (BERT reference embeddings)
- Generate `confusion_matrix.png` and `feature_importance.png`

---

### Component 3: Rewritten Scorer

#### [MODIFY] [candidate_scorer.py](file:///e:/hireflow-ai/backend/candidate_scorer.py)

Key changes:
1. **Configurable `SCORING_WEIGHTS` dict** at the top — editable anytime
2. **Configurable `STRONG_THRESHOLD`** (default 0.6) — editable anytime
3. **Load new model** (`model.pkl` + `feature_config.pkl`) at startup
4. **Sentence-BERT integration** for:
   - Semantic skill matching (resume skills vs JD)
   - Certificate relevance scoring (each cert vs JD)
5. **Feature extraction pipeline** identical to training
6. **`model.predict_proba()`** outputs confidence score (0–100%)
7. **Fallback** to keyword matching if model files don't exist

---

### Component 4: Updated App

#### [MODIFY] [app.py](file:///e:/hireflow-ai/backend/app.py)

- Remove imports of `mock_s3` and `database`
- Remove `init_db()` call
- Remove `save_candidates()` call (we keep JSON storage only)
- Update `/results` endpoint to read from JSON instead of DB
- Keep `/json-data` endpoint as-is

---

### Component 5: Updated Dependencies

#### [MODIFY] [requirements.txt](file:///e:/hireflow-ai/backend/requirements.txt)

```
flask
flask-cors
PyMuPDF
python-docx
werkzeug
scikit-learn
gunicorn
easyocr
Pillow
mammoth
numpy
sentence-transformers
pandas
openpyxl
joblib
```

Removed: `sqlalchemy`, `pymysql`, `cryptography` (not needed without DB layer)

---

## Verification Plan

### Automated Tests
1. **Train the model**: `python train_model.py` from `backend/`
   - Verify no errors
   - Verify `model.pkl` and `feature_config.pkl` are created
   - Verify accuracy is printed and > 70%
   - Verify plots are generated

2. **Test scoring pipeline**: Start Flask, upload a resume, verify:
   - Scores vary between strong and weak candidates
   - A "React Developer" resume scores well against a "Frontend Developer" JD (semantic matching works)
   - Certificates are only scored if relevant to the JD

### Manual Verification
- Ask you to upload resumes via the frontend and check rankings make sense
