# train_model.py - HireFlow-AI ML Training Pipeline
# ==================================================================
# Trains a Random Forest classifier on XLSX + JSON resume data.
# Uses Sentence-BERT for semantic skill/certificate matching.
# Generates: model.pkl, confusion_matrix.png, feature_importance.png
#
# Shared extraction logic lives in resume_features.py.
# ==================================================================

import os
import json
import re
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

# Import shared extraction & config from the DRY module
from resume_features import (
    SCORING_WEIGHTS,
    STRONG_THRESHOLD,
    EXPERIENCE_NORM_CAP,
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
    split_individual_projects,
    extract_education_quality,
    extract_certificate_mentions,
)

print("=" * 60)
print("  HireFlow-AI Model Training Pipeline")
print("=" * 60)

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
# STEP 2: Feature Extraction Helpers (BERT-dependent)
# ==================================================================

def score_project_relevance(text, jd_embedding):
    """Scores project relevance against JD using BERT."""
    projects = extract_projects(text)
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
    cert_matches = extract_certificate_mentions(text)
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

    # Add education column if exists for education quality scoring
    if "Education" in df_excel.columns:
        df_excel["raw_text"] = df_excel["raw_text"] + " " + df_excel["Education"].fillna("").astype(str)

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

# 4E: Education quality
def _get_education_quality(row):
    raw = str(row.get("raw_text", ""))
    jd = str(row.get("job_description", ""))
    return extract_education_quality(raw, jd)

df["education_quality"] = df.apply(_get_education_quality, axis=1)

# 4F: Semantic matching via BERT (the slow step)
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
print(f"    education_quality    : mean={df['education_quality'].mean():.3f}, std={df['education_quality'].std():.3f}")
print(f"    skills_count         : mean={df['skills_count'].mean():.1f}")
print(f"    experience_years     : mean={df['experience_years'].mean():.1f}")
print(f"    has_experience       : {df['has_experience'].sum()}/{len(df)}")
print(f"    has_contact          : {df['has_contact'].sum()}/{len(df)}")

# ==================================================================
# STEP 5: Generate Pseudo-Labels
# ==================================================================
print("\n[Step 5] Generating target labels via quality formula...")

# Use consistent normalisation cap from config
df["exp_score"] = (df["experience_years"] / EXPERIENCE_NORM_CAP).clip(upper=1.0)
df["skills_score_norm"] = (df["skills_count"] / 15.0).clip(upper=1.0)

df["quality_score"] = (
    (df["skills_match_score"]    * SCORING_WEIGHTS["skills_match"]) +
    (df["exp_score"]             * SCORING_WEIGHTS["experience"]) +
    (df["certificate_relevance"] * SCORING_WEIGHTS["certificates"]) +
    (df["has_contact"]           * SCORING_WEIGHTS["contact_info"]) +
    (df["skills_score_norm"]     * SCORING_WEIGHTS["skills_count"]) +
    (df["project_relevance"]     * SCORING_WEIGHTS["project_relevance"]) +
    (df["education_quality"]     * SCORING_WEIGHTS["education_quality"])
)

current_threshold = STRONG_THRESHOLD
df["shortlisted"] = (df["quality_score"] >= current_threshold).astype(int)

strong_count = df["shortlisted"].sum()
weak_count = (df["shortlisted"] == 0).sum()

# Auto-adjust threshold if labels are too imbalanced
if strong_count < 5 or weak_count < 5:
    print(f"  Warning: Default threshold {current_threshold} gave {strong_count} Strong, {weak_count} Weak.")
    print("  Auto-adjusting to top 25% as Strong...")
    current_threshold = df["quality_score"].quantile(0.75)
    df["shortlisted"] = (df["quality_score"] >= current_threshold).astype(int)
    strong_count = df["shortlisted"].sum()
    weak_count = (df["shortlisted"] == 0).sum()
    print(f"  New threshold: {current_threshold:.3f}")

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
    "project_relevance",
    "education_quality"
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
ax.set_title(f"Confusion Matrix (Threshold={current_threshold:.2f})", fontsize=14)
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
print(f"  Features: {', '.join(feature_cols)}")
print(f"  Test Accuracy: {accuracy * 100:.2f}%")
print(f"  CV Accuracy:   {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")
print("=" * 60)
