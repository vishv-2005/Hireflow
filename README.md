# HireFlow AI

> **AI-Powered Resume Screening & Ranking Pipeline** — Upload resumes, score candidates with ML, and find the right talent faster using semantic matching and predictive analytics.

Built by a team of CS students as a high-performance recruitment analytics platform.

---

## 🚀 Key Features (V2 Optimized)

- **Hybrid ML Scoring**: Combines deterministic keyword matching with a **Random Forest** classifier and **Sentence-BERT** semantic similarity for 95% ranking accuracy.
- **Multi-Format OCR Pipeline**: high-accuracy text extraction from **PDF, DOCX, DOC, and Images** (PNG, JPG) using **EasyOCR**.
- **Education Quality Scoring**: Automatically evaluates degree levels (PhD to Diploma) and field-of-study relevance to the job description.
- **Project Relevance Analysis**: Uses NLP to score candidate projects specifically against JD requirements.
- **Anomaly Detection**: Statistical flagging of "keyword stuffing" or abnormally padded resumes.
- **DRY Architecture**: Optimized backend with shared feature extraction for both real-time scoring and batch training.

---

## 🛠️ Tech Stack

### Frontend
- **React 18** (Vite)
- **Vanilla CSS** (Custom Design System)
- **React Router** & **Axios**

### Backend
- **Python 3.10** & **Flask**
- **Machine Learning**: `scikit-learn` (Random Forest), `sentence-transformers` (BERT)
- **NLP**: Semantic cosine similarity for project and skill matching.
- **Parsing**: `PyMuPDF`, `python-docx`, `mammoth`
- **OCR**: `EasyOCR` (GPU/CPU support)
- **Storage**: JSON-based persistent "Data Lake" for ML training history.

---

## 📊 Performance Metrics
The system is trained on a combined dataset of 1,100+ resumes.
- **Test Accuracy**: 94.92%
- **Cross-Validation Accuracy**: 96.43% (5-fold)
- **Inference Speed**: ~50ms per resume (with BERT JD caching)

---

## ⚙️ How to Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- [Sentence-BERT models](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (auto-downloaded on first run)

### Quick Start
1. **Clone & Install**
   ```bash
   git clone https://github.com/your-username/hireflow-ai.git
   cd hireflow-ai
   ```

2. **Unified Launcher**
   Use the root-level launcher to start both services at once:
   ```bash
   python start.py
   ```

3. **Manual Startup**
   - **Backend**: `cd backend && pip install -r requirements.txt && python app.py` (Starts on port 5001)
   - **Frontend**: `cd frontend && npm install && npm run dev` (Starts on port 5173)

---

## 📂 Project Structure

```text
hireflow-ai/
├── backend/
│   ├── app.py              # Flask API & Route Orchestration
│   ├── resume_features.py  # [DRY] Centralized Extraction & Scoring Logic
│   ├── candidate_scorer.py # ML Inference & Hybrid Scoring Pipeline
│   ├── train_model.py      # ML Training Pipeline (Random Forest + BERT)
│   ├── resume_parser.py    # Multi-format Text Extraction & OCR
│   ├── json_storage.py     # Persistent JSON Data Lake
│   ├── model.pkl           # Trained ML Model (Artifact)
│   └── requirements.txt    # Optimized Dependencies
├── frontend/
│   ├── src/
│   │   ├── pages/          # Dashboard, Upload, Landing
│   │   ├── components/     # UI Components
│   │   └── styles/         # Custom CSS
│   └── vite.config.js
├── start.py                # Unified Backend/Frontend Launcher
└── Super_Resume_Dataset_Rows_1_to_1000.xlsx  # Core Training Data
```

---

*HireFlow AI © 2026*
