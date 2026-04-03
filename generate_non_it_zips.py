import os
import random
import zipfile
import shutil
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Sample data for NON-IT generation
FIRST_NAMES = ["Oliver", "Emma", "Liam", "Ava", "Noah", "Sophia", "Lucas", "Isabella", "Mason", "Mia"]
LAST_NAMES = ["Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris"]

# Focus on Pharma and Mechanical
JOB_TYPES = ["Pharma", "Mechanical"]

PHARMA_TITLES = ["Clinical Research Associate", "Pharmacologist", "Regulatory Affairs Specialist", "Quality Control Analyst"]
PHARMA_SKILLS = ["Clinical Trials", "FDA Regulations", "GLP", "GMP", "Pharmacovigilance", "Bioinformatics", "Data Analysis", "Drug Development", "Quality Assurance", "SOP Development"]
PHARMA_DEGREES = ["B.S. in Pharmacy", "M.S. in Pharmacology", "Ph.D. in Biochemistry"]

MECHANICAL_TITLES = ["Mechanical Engineer", "HVAC Specialist", "Manufacturing Engineer", "Robotics Technician"]
MECHANICAL_SKILLS = ["AutoCAD", "SolidWorks", "Thermodynamics", "Fluid Mechanics", "CAD/CAM", "Finite Element Analysis", "HVAC System Design", "Robotics", "Six Sigma", "Project Management"]
MECHANICAL_DEGREES = ["B.S. in Mechanical Engineering", "M.S. in Aerospace Engineering", "B.S. in Industrial Engineering"]

COMPANIES = ["Global Health Corp", "MedLife Industries", "AeroDynamics Inc", "Precision Manufacturing", "BioCure Labs", "FutureMech Systems"]

def generate_non_it_resume_content(candidate_id, job_type):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    
    if job_type == "Pharma":
        title = random.choice(PHARMA_TITLES)
        skills_pool = PHARMA_SKILLS
        degree = random.choice(PHARMA_DEGREES)
        cert = random.choice(["Certified Clinical Research Professional (CCRP)", "Regulatory Affairs Certification", "Good Clinical Practice (GCP) Certification", "None"])
    else:
        title = random.choice(MECHANICAL_TITLES)
        skills_pool = MECHANICAL_SKILLS
        degree = random.choice(MECHANICAL_DEGREES)
        cert = random.choice(["Certified SolidWorks Professional", "Six Sigma Green Belt Certificate", "AutoCAD Certification", "None"])

    phone = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    email = f"{name.replace(' ', '.').lower()}@example.com"
    
    num_skills = random.randint(4, 8)
    skills = random.sample(skills_pool, num_skills)
    
    start_year = random.randint(2012, 2020)
    end_year = random.choice(["Present", "2023", "2024"])
    experience_years = random.randint(3, 12)
    
    content = [
        f"Resume_{candidate_id}",
        f"Name: {name}",
        f"Role: {title}",
        f"Contact: {email} | {phone} | linkedin.com/in/{name.replace(' ', '').lower()}",
        "",
        "SUMMARY",
        f"Dedicated {title} with {experience_years} years of experience.",
        "Committed to high standards of safety, quality, and performance.",
        "",
        "SKILLS",
        ", ".join(skills),
        "",
        "EXPERIENCE",
        f"{random.choice(COMPANIES)}",
        f"{title} ({start_year} - {end_year})",
        "- Led projects aligning with strict industry compliance and guidelines.",
        "- Improved operational efficiency by 15% through workflow optimization.",
        "- Drafted comprehensive reports and analysis for stakeholder review.",
        "",
        "EDUCATION",
        f"{degree} - State University",
        f"Graduated: {start_year - 1}",
        "",
        "CERTIFICATIONS",
        cert
    ]
    return name, "\n".join(content)

def create_pdf(filepath, content):
    c = canvas.Canvas(filepath, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in content.split('\n'):
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def create_docx(filepath, content):
    doc = Document()
    for line in content.split('\n'):
        doc.add_paragraph(line)
    doc.save(filepath)

def generate_non_it_test_zips():
    base_dir = "non_it_test_batches"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    candidate_counter = 1
    
    # Generate 1 zip for Pharma, 1 zip for Mechanical, 1 Mixed
    scenarios = ["Pharma", "Mechanical", "Mixed"]
    
    for zip_idx, scenario in enumerate(scenarios, 1):
        batch_folder = os.path.join(base_dir, f"{scenario.lower()}_batch")
        os.makedirs(batch_folder)
        
        for res_idx in range(1, 11): # 10 resumes each
            if scenario == "Mixed":
                job_type = random.choice(["Pharma", "Mechanical"])
            else:
                job_type = scenario
                
            name, content = generate_non_it_resume_content(candidate_counter, job_type)
            safe_name = name.replace(' ', '_').lower()
            
            # Simple 50/50 split between PDF and DOCX
            if random.random() < 0.5:
                filepath = os.path.join(batch_folder, f"{safe_name}_{candidate_counter}.pdf")
                create_pdf(filepath, content)
            else:
                filepath = os.path.join(batch_folder, f"{safe_name}_{candidate_counter}.docx")
                create_docx(filepath, content)
                
            candidate_counter += 1
            
        zip_filepath = os.path.join(base_dir, f"{scenario.lower()}_batch.zip")
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(batch_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=file)
                    
        print(f"Created {zip_filepath} with 10 {scenario} resumes.")

if __name__ == "__main__":
    generate_non_it_test_zips()
