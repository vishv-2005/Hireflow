import joblib
import pandas as pd
import numpy as np

try:
    model = joblib.load('model.pkl')
    print("Model classes:", model.classes_)
    
    # 6 features
    feature_cols = [
        "skills_match_score",
        "skills_count",
        "has_experience",
        "certificate_relevance",
        "has_contact",
        "experience_years"
    ]
    
    # Let's create a very strong candidate
    X = pd.DataFrame([[0.9, 20, 1, 0.8, 1, 10]], columns=feature_cols)
    print("Strong candidate predictability:")
    proba = model.predict_proba(X)
    print(proba)
    
    # Let's create an average candidate
    X2 = pd.DataFrame([[0.5, 5, 1, 0.0, 1, 3]], columns=feature_cols)
    print("Average candidate predictability:")
    proba2 = model.predict_proba(X2)
    print(proba2)

except Exception as e:
    print("Error:", e)
