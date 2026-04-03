import os
import random
import zipfile
import shutil
from docx import Document
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image, ImageDraw, ImageFont

# Sample data for generation
FIRST_NAMES = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Victor", "Peggy", "Sybil"]
LAST_NAMES = ["Smith", "Doe", "Johnson", "Brown", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson"]
TITLES = ["Software Engineer", "Data Scientist", "Frontend Developer", "Backend Developer", "Machine Learning Engineer", "DevOps Engineer", "Full Stack Developer"]
TECHNICAL_SKILLS = ["Python", "Java", "JavaScript", "SQL", "React", "Machine Learning", "Data Analysis", "AWS", "Docker", "Kubernetes", "Git", "HTML", "CSS", "Node.js", "Flask", "TensorFlow", "Pandas", "MongoDB", "REST API", "Agile", "C++", "C#", "Linux", "GCP", "Azure"]
UNIVERSITIES = ["State University", "Tech Institute", "Global University", "City College", "Metropolitan Institute", "National University"]
COMPANIES = ["TechCorp", "Innovate LLC", "Global Solutions", "NextGen Systems", "Cloud Networks", "DataMinds", "WebWorks"]

def generate_resume_content(candidate_id):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    title = random.choice(TITLES)
    phone = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    email = f"{name.replace(' ', '.').lower()}@example.com"
    
    num_skills = random.randint(3, 12)
    skills = random.sample(TECHNICAL_SKILLS, num_skills)
    
    start_year = random.randint(2010, 2022)
    end_year = random.choice(["Present", "2023", "2024"])
    
    experience_years = random.randint(1, 15)
    
    content = [
        f"Resume_{candidate_id}",
        f"Name: {name}",
        f"Role: {title}",
        f"Contact: {email} | {phone} | linkedin.com/in/{name.replace(' ', '').lower()}",
        "",
        "SUMMARY",
        f"Experienced {title} with {experience_years} years of experience building scalable systems.",
        "Passionate about writing clean, maintainable code and solving complex problems.",
        "",
        "SKILLS",
        ", ".join(skills),
        "",
        "EXPERIENCE",
        f"{random.choice(COMPANIES)}",
        f"{title} ({start_year} - {end_year})",
        "- Developed and maintained web applications.",
        "- Collaborated with cross-functional teams to deliver high-quality software.",
        "- Optimized backend services for performance and scalability.",
        "",
        "EDUCATION",
        f"B.S. in Computer Science - {random.choice(UNIVERSITIES)}",
        f"Graduated: {start_year - 1}",
        "",
        "CERTIFICATIONS",
        random.choice(["AWS Certified Solutions Architect", "Google Cloud Associate Engineer", "Coursera Deep Learning Specialization", "Kubernetes Administrator", "None"])
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

def create_image(filepath, content):
    img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    y = 50
    for line in content.split('\n'):
        d.text((50, y), line, font=font, fill=(0, 0, 0))
        y += 20
    img.save(filepath)

def generate_test_zips(num_zips=5, num_resumes_per_zip=10):
    base_dir = "test_batches"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    candidate_counter = 1
    
    for zip_idx in range(1, num_zips + 1):
        batch_folder = os.path.join(base_dir, f"batch_{zip_idx}")
        os.makedirs(batch_folder)
        
        for res_idx in range(1, num_resumes_per_zip + 1):
            name, content = generate_resume_content(candidate_counter)
            safe_name = name.replace(' ', '_').lower()
            
            # Determine format
            # Let's do roughly 50% PDF, 30% DOCX, 20% Image
            rand_val = random.random()
            if rand_val < 0.5:
                filepath = os.path.join(batch_folder, f"{safe_name}_{candidate_counter}.pdf")
                create_pdf(filepath, content)
            elif rand_val < 0.8:
                filepath = os.path.join(batch_folder, f"{safe_name}_{candidate_counter}.docx")
                create_docx(filepath, content)
            else:
                ext = random.choice([".png", ".jpg"])
                filepath = os.path.join(batch_folder, f"{safe_name}_{candidate_counter}{ext}")
                create_image(filepath, content)
                
            candidate_counter += 1
            
        # Zip the folder
        zip_filename = f"test_batch_{zip_idx}.zip"
        zip_filepath = os.path.join(base_dir, zip_filename)
        
        with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(batch_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add to zip relative to the batch folder
                    zipf.write(file_path, arcname=file)
                    
        print(f"Created {zip_filepath} with {num_resumes_per_zip} resumes.")

if __name__ == "__main__":
    generate_test_zips()
    print("Done generating 5 test zip files in 'test_batches' directory.")
