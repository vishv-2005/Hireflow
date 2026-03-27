# HireFlow AI

> B2B SaaS Big Data recruitment analytics platform — upload resumes, score candidates, and find the right talent faster.

Built by a team of 4 CS students as part of our 6th semester Major Studio Project.

---

## Tech Stack

**Frontend**
- React 18 (with Vite)
- Plain CSS (no frameworks)
- React Router for navigation
- Axios for API calls

**Backend**
- Python 3.10
- Flask + Flask-CORS
- PyMuPDF (`fitz`) for PDF text extraction
- `python-docx` for DOCX parsing
- `mammoth` for legacy DOC parsing
- `pytesseract` + `Pillow` for image OCR (scanned resumes)
- `scikit-learn` for TF-IDF based resume–job matching

**Infrastructure (Coming Soon)**
- AWS S3 for file storage
- AWS SageMaker for ML scoring
- AWS RDS MySQL for persistent data

---

## How to Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip and npm installed
- **Tesseract OCR** installed (required for image resume parsing):
  - **Windows**: `choco install tesseract` or [download installer](https://github.com/UB-Mannheim/tesseract/wiki)
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt-get install tesseract-ocr`

### Step 1 — Clone the Repo
```bash
git clone https://github.com/your-username/hireflow-ai.git
cd hireflow-ai
```

### Step 2 — Start the Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
Flask will start on `http://localhost:5001`

### Step 3 — Start the Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```
Vite dev server will start on `http://localhost:5173`

### Step 4 — Open the App
Go to [http://localhost:5173](http://localhost:5173) in your browser.

---

## What Works Right Now

### Operation 1 — Resume Upload & Parsing (Multi-Format)
- Upload a **ZIP file** containing resumes, or select **multiple individual files**
- Supports **PDF** (via PyMuPDF), **DOCX** (via python-docx), **DOC** (via mammoth), and **image resumes** like PNG, JPG, TIFF, BMP (via Tesseract OCR)
- Mixed formats work together — a ZIP can contain PDFs, Word docs, and scanned images all at once

### Operation 2 — Candidate Scoring & Ranking
- Each resume is scored using **TF-IDF cosine similarity** against a job description (if provided)
- Falls back to keyword matching against 20 common tech skills when no job description is given
- Candidates are ranked by score (highest first)
- Results displayed on a dashboard with color-coded score badges and matched skill tags

### Operation 3 — Anomaly Detection
- Batch-level anomaly detection flags resumes with statistically abnormal text length
- Helps catch keyword stuffing and suspicious resume padding

---

## What's Coming Next

- [ ] AWS S3 integration for file storage
- [ ] AWS SageMaker for ML-based candidate scoring
- [ ] AWS RDS MySQL for persistent data storage
- [ ] User authentication (login/signup)
- [ ] Job description matching improvements
- [ ] Resume preview on dashboard

---

## GitHub Branch Strategy

We follow a simple branching model:

| Branch | Purpose |
|--------|---------|
| `main` | Stable, working code only |
| `dev` | Integration branch for testing |
| `feature/frontend-landing` | Teammate 1 — landing page UI |
| `feature/backend-parser` | Teammate 2 — resume parsing logic |
| `feature/dashboard-ui` | Teammate 3 — dashboard & results |
| `feature/scoring-logic` | Teammate 4 — scoring algorithm |

Each team member should work on their own branch and open a PR to merge into `main`.

---

## Project Structure

```
hireflow-ai/
├── frontend/
│   ├── src/
│   │   ├── pages/          # LandingPage, Dashboard, UploadPage
│   │   ├── components/     # Navbar, FeatureCard, CandidateTable
│   │   ├── styles/         # global.css, landing.css, dashboard.css, upload.css
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── backend/
│   ├── app.py              # Flask routes (ZIP + multi-file upload)
│   ├── resume_parser.py    # PDF/DOCX/DOC/image text extraction
│   ├── candidate_scorer.py # TF-IDF scoring and ranking logic
│   ├── database.py         # SQLAlchemy ORM for candidate storage
│   ├── mock_s3.py          # simulated S3 upload
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

*HireFlow AI © 2025*
