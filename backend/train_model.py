# train_model.py
# This script trains a Random Forest model to predict if a candidate should be shortlisted
# We switched from Decision Tree to Random Forest because the DT was overfitting
# (100% accuracy on training data is way too suspicious lol)
# Random Forest is basically many Decision Trees voting together -- more reliable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # switching from DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Starting model training (Random Forest edition)...")
print("=" * 50)

# ============================================================
# STEP 1: Load the dataset
# 1000 rows of resume data from our college dataset repo
# ============================================================
print("Step 1: Loading the Excel file...")
df = pd.read_excel("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx")
print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2: Drop columns we don't need
# Personal info doesn't help predict shortlisting
# Text-heavy columns are hard to encode cleanly so we drop those too
# ============================================================
print("\nStep 2: Dropping unnecessary columns...")
cols_to_drop = [
    "Name", "Email", "Phone", "City", "Gender",
    "Career_Objective", "Education_Institute",
    "Passing_Year", "Responsibility", "Certifications"
]
df = df.drop(columns=cols_to_drop)
print(f"  Remaining columns: {list(df.columns)}")

# ============================================================
# STEP 3: Handle missing values
# Fill Skills with empty string first so we don't lose those rows
# Then drop whatever else is still null
# The order matters here -- we figured that out the hard way
# ============================================================
print("\nStep 3: Handling missing values...")
df["Skills"] = df["Skills"].fillna("")
before_drop = len(df)
df = df.dropna()
after_drop = len(df)
print(f"  Rows before dropna: {before_drop}")
print(f"  Rows after dropna:  {after_drop}")
print(f"  Dropped {before_drop - after_drop} rows with missing data")

# ============================================================
# STEP 4: Feature Engineering -- skills_count
# The Skills column looks like: ['Python', 'SQL', 'React']
# It's stored as a string, not an actual list -- annoying!
# ast.literal_eval safely converts the string into a real list
# ============================================================
print("\nStep 4: Engineering 'skills_count' feature...")

def count_skills(skills_str):
    # this took us a while to figure out
    # you can't just do len(skills_str) because it's a string, not a list
    try:
        if isinstance(skills_str, list):
            return len(skills_str)
        skills_str = str(skills_str).strip()
        if skills_str == "" or skills_str == "[]":
            return 0
        skills_list = ast.literal_eval(skills_str)
        return len(skills_list)
    except Exception:
        # fallback if ast.literal_eval fails for some weird input
        return len([s for s in skills_str.split(",") if s.strip()])

df["skills_count"] = df["Skills"].apply(count_skills)
print(f"  Average skills per candidate: {df['skills_count'].mean():.2f}")
print(f"  Max skills in one resume:     {df['skills_count'].max()}")

# ============================================================
# STEP 5: Create the target variable -- shortlisted
# Shortlisted = 1 if salary above median AND experience > 5 years
# This is what HRs roughly consider when filtering candidates
# ============================================================
print("\nStep 5: Creating 'shortlisted' target variable...")
median_salary = df["Expected_Salary"].median()
print(f"  Median Expected Salary: {median_salary}")

df["shortlisted"] = (
    (df["Expected_Salary"] > median_salary) &
    (df["Experience_Years"] > 5)
).astype(int)

print(f"  Shortlisted (1): {df['shortlisted'].sum()}")
print(f"  Not shortlisted (0): {(df['shortlisted'] == 0).sum()}")

# ============================================================
# STEP 6: Encode Department and JobRole
# ML models need numbers not strings
# LabelEncoder assigns an integer to each unique category
# We save the encoders so candidate_scorer.py can reuse them
# ============================================================
print("\nStep 6: Label encoding Department and JobRole...")

label_encoders = {}

le_dept = LabelEncoder()
df["Department"] = le_dept.fit_transform(df["Department"])
label_encoders["Department"] = le_dept
print(f"  Department classes: {list(le_dept.classes_)}")

le_role = LabelEncoder()
df["JobRole"] = le_role.fit_transform(df["JobRole"])
label_encoders["JobRole"] = le_role
print(f"  JobRole classes: {list(le_role.classes_)}")

joblib.dump(label_encoders, "label_encoders.pkl")
print("  Saved label_encoders.pkl")

# ============================================================
# STEP 7: Define features (X) and target (y)
# Using 5 features this time -- including Expected_Salary
# because it actually carries real signal here
# ============================================================
print("\nStep 7: Selecting features and target...")

feature_cols = ["Experience_Years", "skills_count", "Expected_Salary", "Department", "JobRole"]
X = df[feature_cols]
y = df["shortlisted"]

print(f"  Features: {feature_cols}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================
# STEP 8: Train / test split
# 80% training, 20% testing
# random_state=42 so results are reproducible every time we run
# ============================================================
print("\nStep 8: Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# ============================================================
# STEP 9: Train the Random Forest Classifier
# Switching to random forest because the decision tree was a bit
# unstable and overfitting on our small dataset (got 100% which is sus)
# Random Forest = 100 different decision trees, each trained on a
# random subset of data and features, then they vote on the answer
# Much more stable and generalises better
# ============================================================
print("\nStep 9: Training Random Forest Classifier...")
print("  (this takes a few seconds -- training 100 trees)")

model = RandomForestClassifier(
    n_estimators=100,   # 100 trees in the forest
    max_depth=8,        # each tree can go 8 levels deep -- prevents overfitting
    random_state=42     # for reproducibility
)
model.fit(X_train, y_train)
print("  Training complete!")

# ============================================================
# STEP 10: Evaluate the model
# Accuracy: overall % of correct predictions
# Classification report: precision, recall, F1 for each class
# ============================================================
print("\nStep 10: Evaluating model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Shortlisted", "Shortlisted"]))

# ============================================================
# STEP 11: Save confusion matrix plot
# Shows where the model made mistakes
# Rows = actual label, Columns = predicted label
# ============================================================
print("Step 11: Saving confusion matrix plot...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix - Random Forest", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Shortlisted", "Shortlisted"])
ax.set_yticklabels(["Not Shortlisted", "Shortlisted"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
print("  Saved confusion_matrix.png")

# ============================================================
# STEP 12: Save feature importance plot
# Random Forest gives us an importance score for each feature
# This tells us which features the model relied on most
# ============================================================
print("Step 12: Saving feature importance plot...")

importances = model.feature_importances_
sorted_idx = np.argsort(importances)
sorted_features = [feature_cols[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(sorted_features, sorted_importances, color="#4f86c6")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Random Forest", fontsize=14)

for bar, val in zip(bars, sorted_importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()
print("  Saved feature_importance.png")

# ============================================================
# STEP 13: Save the model
# This overwrites the old decision tree model.pkl
# ============================================================
print("\nStep 13: Saving model to disk...")
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

print("\n" + "=" * 50)
print("All done! Files saved:")
print("  - model.pkl              (Random Forest, 100 trees)")
print("  - label_encoders.pkl     (Department + JobRole encoders)")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("=" * 50)
