import re

def _split_individual_projects(project_section_text):
    lines = project_section_text.split('\n')
    projects = []
    current_project = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_project:
                proj_text = '\n'.join(current_project)
                if len(proj_text.strip()) > 15:
                    projects.append(proj_text)
                current_project = []
            continue
        
        # Extended bullet characters incl. corrupted ones like ò, and _
        bullet_pattern = r'^(?:[\u2022\u2023\u25E6\u2043\u2219•●○◦►▸▹\-\*ò_~]\s*|\d+[\.\)]\s+|(?:project\s*(?:\d+|[a-z])?\s*[:–\-])|(?:title\s*[:–\-]))'
        is_bullet = bool(re.match(bullet_pattern, stripped, re.IGNORECASE))
        
        # New pattern: does it contain a date range on this line?
        has_date_range = bool(re.search(r'(?:20[0-2][0-9]\s*[-to–]+\s*(?:20[0-2][0-9]|present|now|current|developing))', stripped, re.IGNORECASE))
        
        # New pattern: if previous line was short and this line starts a paragraph?
        
        is_new_project = is_bullet or has_date_range
        
        if is_new_project and current_project:
            proj_text = '\n'.join(current_project)
            if len(proj_text.strip()) > 15:
                projects.append(proj_text)
            current_project = [line]
        else:
            current_project.append(line)
            
    if current_project:
        proj_text = '\n'.join(current_project)
        if len(proj_text.strip()) > 15:
            projects.append(proj_text)
            
    # Fallback chunking if we STILL only have 1 giant project
    if len(projects) <= 1 and len(project_section_text.strip()) > 300:
        chunks = []
        curr = []
        for line in lines:
            if not line.strip(): continue
            curr.append(line)
            # 5 lines or ~300 chars per chunk
            if sum(len(l) for l in curr) > 300:
                chunks.append('\n'.join(curr))
                curr = []
        if curr:
            chunks.append('\n'.join(curr))
        return chunks
        
    if not projects and len(project_section_text.strip()) > 15:
        projects = [project_section_text]
        
    return projects

text = """_
AI-Powered WhatsApp CRM System 	August 2025 – Developing
Developing a lightweight, AI-driven Customer Relationship Management (CRM) system integrated with the WhatsApp Business Cloud API to automate customer interactions for small businesses, featuring real-time message categorization, analytics, and a chatbot assistant.
Building a scalable fullstack application using JS Frameworks for the backend, Flutter for an intuitive dashboard, and SupaBase for data storage, enhancing operational efficiency for small-scale entrepreneurs.
Flutter Chess App for Android                                                                        August 2025 – November 2025
Designed and developed a cross-platform chess game mobile application using Flutter and Dart, enabling intuitive, interactive play on Android devices.
Engineered core chess logic—including move validation, check/checkmate, and special rules (castling,       en-passant, pawn promotion).
Implemented a polished UI/UX with scalable SVG graphics for chess pieces, smooth animations, and custom themes, supporting robust game state persistence with local storage.
Utilized Git and GitHub for version control and collaborative development."""

print("Vishv split:")
for i, p in enumerate(_split_individual_projects(text)):
    print(f"[{i+1}]", repr(p[:60]), "...")

text2 = """AtlasAI, Autonomous AI B2B Lead Generation Tool
òAI-Orchestrated Lead Gen: Developed a local, fully automated system that handles the entire lead 
generation and cold outreach pipeline.
òMulti-Agent Architecture: Engineered a set of background workers (Node.js) that scrape Google Maps for 
leads, analyze their websites, and generate tailored cold emails using a local LLM (Ollama).
Tech Stack: Next.js (React), Node.js, Tailwind CSS, Ollama LLM, Puppeteer.
Sight Sync
òIoT-Enabled Assistive Wearable: Prototyped smart glasses integrating a camera module and 
microcontroller (ESP32/​Raspberry Pi) to capture and process real-time visual data for visually impaired 
users."""

print("\nUzair split:")
for i, p in enumerate(_split_individual_projects(text2)):
    print(f"[{i+1}]", repr(p[:60]), "...")

text3 = """E-commerce website
I built a website using react and nodejs
It has a database

Chatbot
I built a chatbot using python
It uses transformers"""

print("\nChunking split (no bullets, no dates):")
for i, p in enumerate(_split_individual_projects(text3)):
    print(f"[{i+1}]", repr(p[:60]), "...")
