import os
import json
import re
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer, util

print("Starting HireFlow-AI Model Training...")
print("=" * 50)

# SCORING CONFIG - MATCHES candidate_scorer.py
SCORING_WEIGHTS = {
    "skills_match":       0.35,   # semantic skill matching vs JD
    "experience":         0.25,   # years of experience
    "certificates":       0.20,   # relevant certificates vs JD
    "contact_info":       0.10,   # has email, phone, linkedin
    "skills_count":       0.10,   # total distinct skills
}
STRONG_THRESHOLD = 0.6

print("Step 1: Loading Sentence-BERT model (this takes a moment)...")
# We use all-MiniLM-L6-v2 which is fast and lightweight
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# ============================================================
# STEP 2: Feature Extraction Helpers
# ============================================================

def count_skills(skills_data):
    if isinstance(skills_data, list):
        return len(skills_data)
    if pd.isna(skills_data):
        return 0
    skills_data = str(skills_data).strip()
    if not skills_data or skills_data == "[]":
        return 0
    try:
        skills_list = ast.literal_eval(skills_data)
        return len(skills_list)
    except:
        return len([s for s in skills_data.split(",") if s.strip()])

def has_contact_info(text):
    text = str(text).lower()
    # matches email or simple url link or phone-like digits (very loose regex)
    has_email = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0

def has_experience_text(text):
    text = str(text).lower()
    has_keywords = bool(re.search(r'(experience|worked at|employed at|years of)', text))
    has_dates = bool(re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now))', text))
    return 1 if (has_keywords or has_dates) else 0

def extract_experience_years(text):
    text = str(text).lower()
    match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', text)
    if match:
        return min(int(match.group(1)), 25) # Cap at 25
    return 0

def get_certificate_relevance(text, jd_embedding):
    """
    Extracts anything that looks like a certificate, then computes semantic
    similarity to the Job Description. Returns a score 0-1.
    """
    text = str(text).lower()
    cert_matches = []
    
    # 1. Grab explicit certification blocks or keywords
    matches = re.finditer(r'((?:awscertified|certified|certification|certificate|coursera|udemy)[^\n.,]*)', text)
    for m in matches:
        cert_matches.append(m.group(1).strip())
        
    if not cert_matches:
        return 0.0
        
    # Get embeddings for all certs
    cert_embeddings = bert_model.encode(cert_matches, convert_to_tensor=True)
    
    # Compute similarity against JD
    cosine_scores = util.cos_sim(cert_embeddings, jd_embedding)
    
    # We take the top scores.
    # Ex: if they have 1 highly relevant cert vs 3 highly relevant certs
    # Summing the top ones bounded by 1.0 is a good metric
    scores = cosine_scores.cpu().numpy().flatten()
    relevant_scores = [s for s in scores if s > 0.3] # Ignore low relevance
    
    if not relevant_scores:
        return 0.0
        
    total_relevance = sum(relevant_scores)
    return min(total_relevance, 1.0)

# ============================================================
# STEP 3: Load Data (XLSX + JSON)
# ============================================================
print("\nStep 3: Loading data...")

# 1. Load XLSX
df_excel = pd.DataFrame()
if os.path.exists("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx"):
    print("  Loading 1000 rows from Excel...")
    df_excel = pd.read_excel("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx")
    
    # Standardize column names into our raw_text features
    # Since Excel doesn't have raw_text, we merge columns to simulate it
    df_excel["raw_text"] = df_excel["Skills"].fillna("") + " " + \
                           (df_excel.get("Certifications", "").fillna("")) + " " + \
                           (df_excel["Experience_Years"].astype(str) + " years experience")
    
    # Simulate a generic IT job description for the excel data
    df_excel["job_description"] = df_excel["JobRole"].fillna("software engineer")
    df_excel["is_excel"] = True
    
# 2. Load JSON
df_json = pd.DataFrame()
json_path = "candidates_data.json"
if os.path.exists(json_path):
    print("  Loading candidates_data.json...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    json_rows = []
    for batch in data.get("batches", []):
        jd = batch.get("job_description", "software engineer")
        for cand in batch.get("candidates", []):
            json_rows.append({
                "filename": cand.get("filename", ""),
                "raw_text": cand.get("raw_text", ""),
                "job_description": jd,
                "skills_list": cand.get("matched_skills", []),
                "is_excel": False
            })
            
    df_json = pd.DataFrame(json_rows)
    # Deduplicate by filename
    if not df_json.empty:
        before = len(df_json)
        df_json = df_json.drop_duplicates(subset=["filename"])
        after = len(df_json)
        print(f"  Deduplicated JSON: {before} -> {after} unique resumes.")

# Merge
df = pd.concat([df_excel, df_json], ignore_index=True)
print(f"  Total combined resumes: {len(df)}")
if len(df) == 0:
    print("Error: No data found. Make sure Excel or JSON file exists.")
    exit(1)

# ============================================================
# STEP 4: Feature Engineering
# ============================================================
print("\nStep 4: Extracting features & semantic embeddings...")

# 1. Base counts & regex extractions
df["skills_count"] = df.apply(lambda row: count_skills(row.get("skills_list", [])) if not row["is_excel"] else count_skills(row["Skills"]), axis=1)
df["has_experience"] = df["raw_text"].apply(has_experience_text)
df["has_contact"] = df["raw_text"].apply(has_contact_info)

# For excel, we already have Experience_Years. For JSON, we regex it.
def get_years(row):
    if row.get("is_excel", False) and "Experience_Years" in row:
        return min(int(row["Experience_Years"]) if pd.notnull(row["Experience_Years"]) else 0, 25)
    return extract_experience_years(row["raw_text"])
df["experience_years"] = df.apply(get_years, axis=1)

# 2. Semantic Matching (Slow step)
print("  Computing semantic similarity against Job Descriptions...")
# We do this efficiently by pre-encoding all JDs and Resumes
jd_texts = df["job_description"].tolist()
resume_texts = df["raw_text"].tolist()

jd_embeddings = bert_model.encode(jd_texts, convert_to_tensor=True)
resume_embeddings = bert_model.encode(resume_texts, convert_to_tensor=True)

# Similarity of resume vs JD
skills_match_scores = []
cert_relevance_scores = []

for i in range(len(df)):
    # Skill match is simply entire resume text vs JD
    sim = util.cos_sim(resume_embeddings[i], jd_embeddings[i]).item()
    # Normalize heavily to 0-1 (BERT cosine values usually between 0.1 and 0.8 here)
    norm_sim = min(max((sim - 0.1) * 1.5, 0.0), 1.0)
    skills_match_scores.append(norm_sim)
    
    # Cert matching
    cert_rel = get_certificate_relevance(df.iloc[i]["raw_text"], jd_embeddings[i])
    cert_relevance_scores.append(cert_rel)

df["skills_match_score"] = skills_match_scores
df["certificate_relevance"] = cert_relevance_scores

# ============================================================
# STEP 5: Generate Pseudo-Labels
# ============================================================
print("\nStep 5: Generating target labels via quality formula...")

# Normalize continuous vars for the formula
df["exp_score"] = df["experience_years"] / 10.0
df["exp_score"] = df["exp_score"].clip(upper=1.0)

df["skills_score"] = df["skills_count"] / 15.0
df["skills_score"] = df["skills_score"].clip(upper=1.0)

df["quality_score"] = (
    (df["skills_match_score"]    * SCORING_WEIGHTS["skills_match"]) +
    (df["exp_score"]             * SCORING_WEIGHTS["experience"]) +
    (df["certificate_relevance"] * SCORING_WEIGHTS["certificates"]) +
    (df["has_contact"]           * SCORING_WEIGHTS["contact_info"]) +
    (df["skills_score"]          * SCORING_WEIGHTS["skills_count"])
)

df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)

if df["shortlisted"].sum() < 5 or (df["shortlisted"] == 0).sum() < 5:
    print("  Warning: Default threshold resulted in too few samples of one class. Auto-adjusting to top 25%.")
    STRONG_THRESHOLD = df["quality_score"].quantile(0.75)
    df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)

print(f"  Distribution: {df['shortlisted'].sum()} Strong (1), {(df['shortlisted']==0).sum()} Weak (0)")

# ============================================================
# STEP 6: Train Random Forest
# ============================================================
print("\nStep 6: Training Random Forest model...")

feature_cols = [
    "skills_match_score", 
    "skills_count", 
    "has_experience", 
    "certificate_relevance", 
    "has_contact", 
    "experience_years"
]
X = df[feature_cols]
y = df["shortlisted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)

# ============================================================
# STEP 7: Evaluate
# ============================================================
print("\nStep 7: Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=["Weak", "Strong"], zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix (Thresh={STRONG_THRESHOLD})", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Weak", "Strong"])
ax.set_yticklabels(["Weak", "Strong"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()

# Feature Importance
importances = model.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = [feature_cols[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(sorted_features, sorted_importances, color="#4f86c6")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Random Forest", fontsize=14)
for bar, val in zip(bars, sorted_importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()

# ============================================================
# STEP 8: Save Model
# ============================================================
print("\nStep 8: Saving assets...")
joblib.dump(model, "model.pkl")

# We don't save BERT reference embeddings because they are dynamic per job description.
# The scorer will load Sentence-BERT on its own.

print("\n" + "=" * 50)
print("Done! Artifacts generated:")
print("  - model.pkl              (Random Forest, 200 trees)")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("=" * 50)
