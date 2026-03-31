# train_model.py
# This script trains a Decision Tree model to predict if a candidate should be shortlisted
# We're using a real dataset of 1000 resumes for this
# Written for our 6th semester project - HireFlow AI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast  # this is needed to safely parse the skills list string

print("Starting model training...")
print("="*50)

# ============================================================
# STEP 1: Load the dataset
# We got this dataset from our college dataset repository
# It has 1000 rows of resume data
# ============================================================
print("Step 1: Loading the Excel file...")
df = pd.read_excel("data/Super_Resume_Dataset_Rows_1_to_1000.xlsx")
print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")
print(f"  Columns: {list(df.columns)}")

# ============================================================
# STEP 2: Drop columns we don't need for prediction
# We're dropping personal info (Name, Email, Phone etc.) 
# because they shouldn't affect whether someone gets shortlisted
# Also dropping text-heavy columns we can't easily encode
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
# Fill Skills with empty string so we don't lose rows
# Drop any row that still has nulls after that
# This took us a while to figure out - order matters here!
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
# STEP 4: Feature Engineering - skills_count
# The Skills column looks like a Python list string e.g. ['Python', 'Java']
# We use ast.literal_eval to safely convert it to an actual list
# Then we count the number of skills -- more skills = better?
# ============================================================
print("\nStep 4: Engineering 'skills_count' feature...")

def count_skills(skills_str):
    # this took us a while to figure out
    # the skills column is stored as a string that LOOKS like a list
    # we need to carefully convert it to an actual list first
    try:
        if isinstance(skills_str, list):
            return len(skills_str)  # already a list, just count it
        skills_str = str(skills_str).strip()
        if skills_str == "" or skills_str == "[]":
            return 0
        skills_list = ast.literal_eval(skills_str)  # safely parse the list string
        return len(skills_list)
    except Exception:
        # if parsing fails for any reason, count comma-separated items as fallback
        return len([s for s in skills_str.split(",") if s.strip()])

df["skills_count"] = df["Skills"].apply(count_skills)
print(f"  Average skills per candidate: {df['skills_count'].mean():.2f}")
print(f"  Max skills in one resume:     {df['skills_count'].max()}")

# ============================================================
# STEP 5: Create the target variable - shortlisted
# A candidate is shortlisted (1) if BOTH conditions are true:
#   - Expected_Salary > median salary in the dataset
#   - Experience_Years > 5
# Otherwise they're not shortlisted (0)
# We came up with this logic based on what HRs usually look for
# ============================================================
print("\nStep 5: Creating 'shortlisted' target variable...")
median_salary = df["Expected_Salary"].median()
print(f"  Median Expected Salary: {median_salary}")

# 1 if both conditions met, 0 otherwise
df["shortlisted"] = (
    (df["Expected_Salary"] > median_salary) & 
    (df["Experience_Years"] > 5)
).astype(int)

shortlisted_count = df["shortlisted"].sum()
not_shortlisted_count = len(df) - shortlisted_count
print(f"  Shortlisted (1): {shortlisted_count}")
print(f"  Not shortlisted (0): {not_shortlisted_count}")

# ============================================================
# STEP 6: Encode categorical columns (Department and JobRole)
# Machine learning models can't handle text directly
# LabelEncoder converts each unique string to a number
# e.g. "Computer Science" -> 0, "Electrical" -> 1, etc.
# We save the encoders so we can use them later during scoring
# ============================================================
print("\nStep 6: Label encoding Department and JobRole...")

label_encoders = {}  # dictionary to store both encoders

le_dept = LabelEncoder()
df["Department"] = le_dept.fit_transform(df["Department"])
label_encoders["Department"] = le_dept
print(f"  Department classes: {list(le_dept.classes_)}")

le_role = LabelEncoder()
df["JobRole"] = le_role.fit_transform(df["JobRole"])
label_encoders["JobRole"] = le_role
print(f"  JobRole classes: {list(le_role.classes_)}")

# save the encoders to a file so candidate_scorer.py can use them later
joblib.dump(label_encoders, "label_encoders.pkl")
print("  Saved label_encoders.pkl")

# ============================================================
# STEP 7: Define features (X) and target (y)
# We use only 4 features -- Expected_Salary is NOT included here
# even though it was used to build the target label above
# including it would be "cheating" (the model would just memorise the salary rule)
# ============================================================
print("\nStep 7: Selecting features and target...")

feature_cols = ["Experience_Years", "skills_count", "Department", "JobRole"]
X = df[feature_cols]
y = df["shortlisted"]

print(f"  Features: {feature_cols}")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# ============================================================
# STEP 8: Split data into training and testing sets
# 80% for training, 20% for testing
# random_state=42 makes it reproducible (we always get same split)
# ============================================================
print("\nStep 8: Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples:  {len(X_test)}")

# ============================================================
# STEP 9: Train the Decision Tree Classifier
# max_depth=5 means the tree won't go too deep (avoids overfitting)
# we tried max_depth=10 but it was overfitting badly
# ============================================================
print("\nStep 9: Training Decision Tree Classifier...")
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("  Training complete!")

# ============================================================
# STEP 10: Evaluate the model
# Accuracy tells us overall how often the model is correct
# Classification report shows precision, recall, F1 per class
# ============================================================
print("\nStep 10: Evaluating model performance...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Shortlisted", "Shortlisted"]))

# ============================================================
# STEP 11: Save confusion matrix as an image
# Confusion matrix shows us where the model made mistakes
# True Positives, False Positives, True Negatives, False Negatives
# ============================================================
print("Step 11: Saving confusion matrix plot...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))

# plot the matrix as a heatmap manually (we didn't want to import seaborn just for this)
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)

# label axes and cells
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix", fontsize=14)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Not Shortlisted", "Shortlisted"])
ax.set_yticklabels(["Not Shortlisted", "Shortlisted"])

# write numbers inside each cell
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
# STEP 12: Save feature importance as an image
# This shows which features the Decision Tree relied on most
# We expected Experience_Years to be important - let's see!
# ============================================================
print("Step 12: Saving feature importance plot...")

importances = model.feature_importances_
feature_names = feature_cols

# sort by importance so the chart looks nicer
sorted_idx = np.argsort(importances)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.barh(sorted_features, sorted_importances, color="#4f86c6")
ax.set_xlabel("Importance Score", fontsize=12)
ax.set_title("Feature Importance - Decision Tree", fontsize=14)

# add value labels to each bar
for bar, val in zip(bars, sorted_importances):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100)
plt.close()
print("  Saved feature_importance.png")

# ============================================================
# STEP 13: Save the trained model
# We use joblib to serialize the model object to a .pkl file
# This way candidate_scorer.py can load it without retraining
# ============================================================
print("\nStep 13: Saving model to disk...")
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

print("\n" + "="*50)
print("All done! Files saved:")
print("  - model.pkl          (the trained Decision Tree)")
print("  - label_encoders.pkl (Department + JobRole encoders)")
print("  - confusion_matrix.png")
print("  - feature_importance.png")
print("="*50)
