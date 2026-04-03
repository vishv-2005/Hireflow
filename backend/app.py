# app.py - main Flask backend for HireFlow AI
# this handles file uploads, resume parsing, and returning ranked results

import os
import zipfile
import shutil
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from resume_parser import parse_resume, ALL_SUPPORTED_EXTENSIONS
from candidate_scorer import score_candidate, rank_candidates, detect_anomalies
from json_storage import save_to_json, load_json_data

app = Flask(__name__)
CORS(app)  # this lets our React frontend talk to Flask without CORS errors

# folders for storing uploaded zips and extracted resumes
if os.environ.get("VERCEL"):
    UPLOAD_FOLDER = "/tmp/uploads"
    EXTRACT_FOLDER = "/tmp/extracted"
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    EXTRACT_FOLDER = os.path.join(os.path.dirname(__file__), "extracted")

# make sure the folders exist when the app starts
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

def is_supported_file(filename):
    """Check if a file has a supported extension for resume parsing."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALL_SUPPORTED_EXTENSIONS

def process_extracted_files(folder_path):
    """Walk through a folder and parse every supported resume file."""
    parsed_resumes = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.startswith(".") or fname.startswith("__"):
                continue
            if not is_supported_file(fname):
                print(f"skipping unsupported file: {fname}")
                continue
            filepath = os.path.join(root, fname)
            print(f"parsing: {fname}")
            result = parse_resume(filepath)
            if result is not None:
                parsed_resumes.append(result)
    return parsed_resumes

@app.route("/upload", methods=["POST"])
def upload_resumes():
    """
    Accepts resumes in two modes: ZIP file or multiple individual files.
    Parses, scores using ML, and returns ranked results as JSON.
    """
    job_description = request.form.get("job_description", "")

    has_zip = "file" in request.files and request.files["file"].filename.endswith(".zip")
    has_files = "files" in request.files

    if not has_zip and not has_files:
        if "file" in request.files:
            single_file = request.files["file"]
            if single_file.filename and is_supported_file(single_file.filename):
                has_files = True
                request.files = {"files": single_file}
            else:
                return jsonify({"error": "No supported files uploaded. Please upload PDF, DOCX, DOC, or image files."}), 400
        else:
            return jsonify({"error": "No file was uploaded."}), 400

    try:
        # clear out any old extracted files
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)

        parsed_resumes = []

        if has_zip:
            # === ZIP MODE ===
            file = request.files["file"]
            print(f"got a ZIP file: {file.filename}, starting to process...")

            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)

            print("unzipping the file...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_FOLDER)

            parsed_resumes = process_extracted_files(EXTRACT_FOLDER)

        else:
            # === MULTI-FILE MODE ===
            files = request.files.getlist("files")
            if not files:
                single = request.files.get("files")
                files = [single] if single else []

            print(f"got {len(files)} individual file(s), starting to process...")

            for f in files:
                if not f or not f.filename:
                    continue
                if not is_supported_file(f.filename):
                    continue

                safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
                filepath = os.path.join(EXTRACT_FOLDER, safe_name)
                f.save(filepath)
                print(f"saved: {f.filename} -> {safe_name}")

                result = parse_resume(filepath)
                if result is not None:
                    result["filename"] = f.filename
                    parsed_resumes.append(result)

        if len(parsed_resumes) == 0:
            return jsonify({"error": "No valid resume files found."}), 400

        print(f"parsed {len(parsed_resumes)} resumes, now scoring...")

        # Score each parsed resume
        scored_candidates = []
        for resume_data in parsed_resumes:
            scored = score_candidate(resume_data, job_description)
            scored_candidates.append(scored)

        # Rank by score highest first
        ranked_candidates = rank_candidates(scored_candidates)

        # Generate unique batch ID
        batch_id = str(uuid.uuid4())

        # Save to JSON data lake FIRST (preserves raw_text for future ML training)
        save_to_json(ranked_candidates, batch_id, job_description)

        # Run anomaly detection (Keyword stuffing check - strips raw_text)
        ranked_candidates = detect_anomalies(ranked_candidates)

        print(f"done! ranked and saved {len(ranked_candidates)} candidates under batch {batch_id}")

        return jsonify({
            "message": f"successfully processed {len(ranked_candidates)} resumes",
            "batch_id": batch_id,
            "count": len(ranked_candidates),
            "candidates": ranked_candidates
        }), 200

    except zipfile.BadZipFile:
        return jsonify({"error": "the uploaded file is not a valid ZIP"}), 400
    except Exception as e:
        print(f"something went wrong: {e}")
        return jsonify({"error": f"server error: {str(e)}"}), 500


@app.route("/results", methods=["GET"])
def get_results():
    """
    Returns the latest batch of processed results directly from the JSON.
    """
    # Fetch from json instead of DB since DB is removed
    data = load_json_data()
    latest_results = []
    if data and data.get("batches"):
        latest_results = data["batches"][-1]["candidates"]
    
    return jsonify({
        "candidates": latest_results,
        "count": len(latest_results)
    }), 200

@app.route("/json-data", methods=["GET"])
def get_json_data():
    """
    Returns the full JSON storage data (all batches).
    """
    data = load_json_data()
    return jsonify(data), 200

if __name__ == "__main__":
    print("starting HireFlow AI backend on http://localhost:5001")
    app.run(debug=True, port=5001)
