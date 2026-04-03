# train_model.py
# Merged version: Prayag's Random Forest + Vishv's Sentence-BERT semantic features
# This trains on both the Excel dataset and any real resumes stored in candidates_data.json
# The model learns to predict "Strong" vs "Weak" candidates based on semantic + structural signals

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

print("Starting HireFlow-AI Model Training (RF + Sentence-BERT)...")
print("=" * 50)

# ============================================================
# SCORING WEIGHTS -- same as candidate_scorer.py
# These determine how much each signal contributes to the label
# Adjust these if you want to prioritise different factors
# ============================================================
SCORING_WEIGHTS = {
    "skills_match":   0.35,  # how semantically similar the resume is to the job role
    "experience":     0.25,  # years of experience
    "certificates":   0.20,  # relevant certifications vs job description
    "contact_info":   0.10,  # has email / phone / linkedin
    "skills_count":   0.10,  # total distinct skills count
}
STRONG_THRESHOLD = 0.6  # quality score >= 60% = shortlisted (class 1)

# ============================================================
# STEP 1: Load Sentence-BERT
# We use the lightweight all-MiniLM-L6-v2 model
# It downloads automatically on first run (~80MB)
# ============================================================
print("\nStep 1: Loading Sentence-BERT model (first run downloads ~80MB)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("  BERT ready!")

# ============================================================
# STEP 2: Feature Extraction Helpers
# These are reused by candidate_scorer.py during inference too
# ============================================================

def count_skills(skills_data):
    """Count skills from either a list or a string like ['Python', 'SQL']."""
    if isinstance(skills_data, list):
        return len(skills_data)
    if pd.isna(skills_data):
        return 0
    skills_data = str(skills_data).strip()
    if not skills_data or skills_data == "[]":
        return 0
    try:
        return len(ast.literal_eval(skills_data))
    except Exception:
        return len([s for s in skills_data.split(",") if s.strip()])

def has_contact_info(text):
    """Returns 1 if resume has email, phone, or LinkedIn -- else 0."""
    text = str(text).lower()
    has_email    = bool(re.search(r'[\w\.-]+@[\w\.-]+', text))
    has_phone    = bool(re.search(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text))
    has_linkedin = 'linkedin.com' in text
    return 1 if (has_email or has_phone or has_linkedin) else 0

def has_experience_text(text):
    """Returns 1 if resume mentions work experience keywords or date ranges."""
    text = str(text).lower()
    has_keywords = bool(re.search(r'(experience|worked at|employed at|years of)', text))
    has_dates    = bool(re.search(r'(20[0-2][0-9]\s*[-–to]+\s*(20[0-2][0-9]|present|now))', text))
    return 1 if (has_keywords or has_dates) else 0

def extract_experience_years(text):
    """Parses 'X years experience' from text. Caps at 25."""
    text = str(text).lower()
    match = re.search(r'(\d+)\s*\+?\s*years?\s*(of\s+)?(experience)?', text)
    return min(int(match.group(1)), 25) if match else 0

def get_certificate_relevance(text, jd_embedding):
    """
    Finds certification mentions in the resume and computes semantic
    similarity against the job description embedding. Returns 0-1.
    """
    text = str(text).lower()
    cert_matches = []
    for m in re.finditer(r'((?:awscertified|certified|certification|certificate|coursera|udemy)[^\n.,]*)', text):
        cert_matches.append(m.group(1).strip())

    if not cert_matches:
        return 0.0

    cert_embeddings = bert_model.encode(cert_matches, convert_to_tensor=True)
    scores = util.cos_sim(cert_embeddings, jd_embedding).cpu().numpy().flatten()
    relevant = [s for s in scores if s > 0.3]
    return min(sum(relevant), 1.0) if relevant else 0.0

# ============================================================
# STEP 3: Load Data
# We combine the Excel dataset with any real JSON resumes
# ============================================================
print("\nStep 3: Loading data...")

df_excel = pd.DataFrame()
if os.path.exists("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx"):
    print("  Loading Excel dataset (1000 rows)...")
    df_excel = pd.read_excel("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx")

    # Build a "raw_text" column by combining relevant fields
    # This mimics what we'd get from a real resume parser
    df_excel["raw_text"] = (
        df_excel["Skills"].fillna("") + " " +
        df_excel.get("Certifications", pd.Series([""] * len(df_excel))).fillna("") + " " +
        df_excel["Experience_Years"].astype(str) + " years experience"
    )
    df_excel["job_description"] = df_excel["JobRole"].fillna("software engineer")
    df_excel["is_excel"] = True

df_json = pd.DataFrame()
json_path = "candidates_data.json"
if os.path.exists(json_path):
    print("  Loading candidates_data.json (real uploaded resumes)...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    json_rows = []
    for batch in data.get("batches", []):
        jd = batch.get("job_description", "software engineer")
        for cand in batch.get("candidates", []):
            json_rows.append({
                "filename":    cand.get("filename", ""),
                "raw_text":    cand.get("raw_text", ""),
                "job_description": jd,
                "skills_list": cand.get("matched_skills", []),
                "is_excel":    False
            })

    df_json = pd.DataFrame(json_rows)
    if not df_json.empty:
        before = len(df_json)
        df_json = df_json.drop_duplicates(subset=["filename"])
        print(f"  JSON resumes (deduplicated): {before} -> {len(df_json)}")

df = pd.concat([df_excel, df_json], ignore_index=True)
print(f"  Total rows for training: {len(df)}")
if len(df) == 0:
    print("ERROR: No data found. Make sure data/Super_Resume_Dataset_Rows_1_to_1000.xlsx exists.")
    exit(1)

# ============================================================
# STEP 4: Feature Engineering
# ============================================================
print("\nStep 4: Extracting features & computing BERT semantic scores...")

# Structural features
df["skills_count"]  = df.apply(
    lambda row: count_skills(row.get("skills_list", [])) if not row["is_excel"] else count_skills(row.get("Skills", [])),
    axis=1
)
df["has_experience"] = df["raw_text"].apply(has_experience_text)
df["has_contact"]    = df["raw_text"].apply(has_contact_info)

def get_years(row):
    if row.get("is_excel", False) and "Experience_Years" in row and pd.notnull(row["Experience_Years"]):
        return min(int(row["Experience_Years"]), 25)
    return extract_experience_years(row["raw_text"])

df["experience_years"] = df.apply(get_years, axis=1)

# Semantic features via Sentence-BERT
print("  Computing semantic similarity (this is the slow step)...")
jd_texts     = df["job_description"].tolist()
resume_texts = df["raw_text"].tolist()

jd_embeddings     = bert_model.encode(jd_texts, convert_to_tensor=True, show_progress_bar=False)
resume_embeddings = bert_model.encode(resume_texts, convert_to_tensor=True, show_progress_bar=False)

skills_match_scores   = []
cert_relevance_scores = []

for i in range(len(df)):
    sim      = util.cos_sim(resume_embeddings[i], jd_embeddings[i]).item()
    norm_sim = min(max((sim - 0.1) * 1.5, 0.0), 1.0)  # normalize BERT scores to 0-1
    skills_match_scores.append(norm_sim)
    cert_relevance_scores.append(get_certificate_relevance(df.iloc[i]["raw_text"], jd_embeddings[i]))

df["skills_match_score"]   = skills_match_scores
df["certificate_relevance"] = cert_relevance_scores
print("  Semantic features done!")

# ============================================================
# STEP 5: Generate Target Labels
# We use the quality formula instead of a hardcoded salary rule
# This is more generalizable across different job types
# ============================================================
print("\nStep 5: Generating shortlisted labels via quality formula...")

df["exp_score"]    = (df["experience_years"] / 10.0).clip(upper=1.0)
df["skills_score"] = (df["skills_count"] / 15.0).clip(upper=1.0)

df["quality_score"] = (
    df["skills_match_score"]    * SCORING_WEIGHTS["skills_match"]  +
    df["exp_score"]             * SCORING_WEIGHTS["experience"]     +
    df["certificate_relevance"] * SCORING_WEIGHTS["certificates"]   +
    df["has_contact"]           * SCORING_WEIGHTS["contact_info"]   +
    df["skills_score"]          * SCORING_WEIGHTS["skills_count"]
)

df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)

# Safety check: if one class has too few samples, auto-adjust threshold
if df["shortlisted"].sum() < 5 or (df["shortlisted"] == 0).sum() < 5:
    print("  Warning: Imbalanced classes after threshold -- auto-adjusting to top 25%.")
    STRONG_THRESHOLD = df["quality_score"].quantile(0.75)
    df["shortlisted"] = (df["quality_score"] >= STRONG_THRESHOLD).astype(int)

print(f"  Shortlisted (1): {df['shortlisted'].sum()}")
print(f"  Not shortlisted (0): {(df['shortlisted'] == 0).sum()}")

# ============================================================
# STEP 6: Train Random Forest
# Features are all semantic + structural signals from above
# No raw salary or department encoding needed -- fully text-driven
# ============================================================
print("\nStep 6: Training Random Forest Classifier (200 trees)...")

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
print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
print("  Training complete!")

# ============================================================
# STEP 7: Evaluate
# ============================================================
print("\nStep 7: Evaluating model...")
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Shortlisted", "Shortlisted"], zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("True", fontsize=12)
ax.set_title(f"Confusion Matrix (Threshold={STRONG_THRESHOLD:.2f})", fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Shortlisted", "Shortlisted"])
ax.set_yticklabels(["Not Shortlisted", "Shortlisted"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
print("  Saved confusion_matrix.png")

# Feature importance plot
importances   = model.feature_importances_
sorted_idx    = np.argsort(importances)
sorted_feats  = [feature_cols[i] for i in sorted_idx]
sorted_imps   = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(sorted_feats, sorted_imps, color="#4f86c6")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Random Forest + BERT", fontsize=14)
for bar, val in zip(bars, sorted_imps):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=10)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()
print("  Saved feature_importance.png")

# ============================================================
# STEP 8: Save Model
# Only the RF model is saved -- BERT is loaded on demand in candidate_scorer.py
# ============================================================
print("\nStep 8: Saving model.pkl...")
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

print("\n" + "=" * 50)
print("Done! Artifacts saved:")
print("  - model.pkl              (Random Forest, 200 trees, BERT features)")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("=" * 50)
