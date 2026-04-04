"""Debug script to compare Ivan Miller and Mallory Anderson scoring."""
from resume_parser import parse_resume
from candidate_scorer import (
    _get_matched_skills, _get_jd_skills, _compute_jd_overlap,
    _has_contact_info, _has_experience_text, _extract_experience_years,
    _count_skills, _load_models, _get_semantic_features, _model, _bert_model,
    SCORING_WEIGHTS
)
import re

JD = "Need a software developer having operation in python, java, javascript, react and wed development. Full stack experience is preferred"

# Parse both resumes
ivan = parse_resume(r'E:\hireflow-ai\test_batches\batch_1\ivan_miller_3.pdf')
mallory = parse_resume(r'E:\hireflow-ai\test_batches\batch_1\mallory_anderson_8.pdf')

print("=" * 70)
print("IVAN MILLER - Raw Text")
print("=" * 70)
print(ivan['raw_text'])
print()
print("=" * 70)
print("MALLORY ANDERSON - Raw Text")
print("=" * 70)
print(mallory['raw_text'])
print()

# Compare features
jd_skills = _get_jd_skills(JD)
import sys
with open("debug_results.log", "w") as f:
    f.write(f"JD Skills extracted: {jd_skills}\n")
    for name, data in [("IVAN", ivan), ("MALLORY", mallory)]:
        raw = data['raw_text']
        matched = _get_matched_skills(raw)
        jd_overlap = _compute_jd_overlap(matched, jd_skills)
        jd_matched = [s for s in matched if s in set(jd_skills)]
        skills_count = _count_skills(matched)
        has_exp = _has_experience_text(raw)
        has_contact = _has_contact_info(raw)
        exp_years = _extract_experience_years(raw)
        
        f.write(f"\n{'=' * 70}\n")
        f.write(f"  {name} - Feature Breakdown\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"  Matched skills ({len(matched)}): {matched}\n")
        f.write(f"  JD matched skills ({len(jd_matched)}): {jd_matched}\n")
        f.write(f"  JD overlap score: {jd_overlap:.3f}\n")
        f.write(f"  Skills count: {skills_count}\n")
        f.write(f"  Has experience text: {has_exp}\n")
        f.write(f"  Has contact info: {has_contact}\n")
        f.write(f"  Experience years: {exp_years}\n")
        
        # Check certs
        cert_patterns = [
            r'((?:aws[\s\-]?certified|certified|certification|certificate|coursera|udemy|google[\s\-]?cloud|azure[\s\-]?certified)[^\n.,;]*)',
        ]
        cert_matches = []
        for pattern in cert_patterns:
            for m in re.finditer(pattern, raw.lower()):
                cert_text = m.group(1).strip()
                if len(cert_text) > 5:
                    cert_matches.append(cert_text)
        f.write(f"  Certificate mentions: {cert_matches}\n")

    # Now score with full ML pipeline
    _load_models()
    if _model is not None and _bert_model is not None:
        for name, data in [("IVAN", ivan), ("MALLORY", mallory)]:
            raw = data['raw_text']
            matched = _get_matched_skills(raw)
            jd_overlap = _compute_jd_overlap(matched, jd_skills)
            skills_count = _count_skills(matched)
            has_exp = _has_experience_text(raw)
            has_contact = _has_contact_info(raw)
            exp_years = _extract_experience_years(raw)
            sm_score, cert_score = _get_semantic_features(raw, JD)
            
            import pandas as pd
            feature_cols = ["skills_match_score", "skills_count", "has_experience",
                            "certificate_relevance", "has_contact", "experience_years"]
            X_infer = pd.DataFrame(
                [[sm_score, skills_count, has_exp, cert_score, has_contact, exp_years]],
                columns=feature_cols
            )
            proba = _model.predict_proba(X_infer)[0]
            
            quality_score = (
                (jd_overlap * SCORING_WEIGHTS["jd_skill_overlap"]) +
                (sm_score * SCORING_WEIGHTS["skills_match"]) +
                (min(exp_years / 10.0, 1.0) * SCORING_WEIGHTS["experience"]) +
                (cert_score * SCORING_WEIGHTS["certificates"]) +
                (has_contact * SCORING_WEIGHTS["contact_info"]) +
                (min(skills_count / 15.0, 1.0) * SCORING_WEIGHTS["skills_count"])
            )
            combined = (quality_score * 0.80) + (float(proba[1]) * 0.20)
            score = round(combined * 100, 1)
            
            f.write(f"\n--- {name} Scoring Breakdown ---\n")
            f.write(f"  JD overlap:     {jd_overlap:.3f} x {SCORING_WEIGHTS['jd_skill_overlap']} = {jd_overlap * SCORING_WEIGHTS['jd_skill_overlap']:.4f}\n")
            f.write(f"  Semantic match: {sm_score:.3f} x {SCORING_WEIGHTS['skills_match']} = {sm_score * SCORING_WEIGHTS['skills_match']:.4f}\n")
            f.write(f"  Experience:     {min(exp_years/10.0, 1.0):.3f} x {SCORING_WEIGHTS['experience']} = {min(exp_years/10.0, 1.0) * SCORING_WEIGHTS['experience']:.4f}\n")
            f.write(f"  Cert relevance: {cert_score:.3f} x {SCORING_WEIGHTS['certificates']} = {cert_score * SCORING_WEIGHTS['certificates']:.4f}\n")
            f.write(f"  Contact info:   {has_contact} x {SCORING_WEIGHTS['contact_info']} = {has_contact * SCORING_WEIGHTS['contact_info']:.4f}\n")
            f.write(f"  Skills count:   {min(skills_count/15.0, 1.0):.3f} x {SCORING_WEIGHTS['skills_count']} = {min(skills_count/15.0, 1.0) * SCORING_WEIGHTS['skills_count']:.4f}\n")
            f.write(f"  --- Quality Score: {quality_score:.4f}\n")
            f.write(f"  ML proba[strong]: {float(proba[1]):.4f}\n")
            f.write(f"  Blend: {quality_score:.4f} * 0.80 + {float(proba[1]):.4f} * 0.20 = {combined:.4f}\n")
            f.write(f"  FINAL SCORE: {score}%\n")
