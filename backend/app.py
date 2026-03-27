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
from mock_s3 import mock_upload_to_s3
from database import init_db, save_candidates, get_latest_batch
from json_storage import save_to_json, load_json_data

app = Flask(__name__)
CORS(app)  # this lets our React frontend talk to Flask without CORS errors

# folders for storing uploaded zips and extracted resumes
# Vercel serverless environments only allow writing to /tmp
if os.environ.get("VERCEL"):
    UPLOAD_FOLDER = "/tmp/uploads"
    EXTRACT_FOLDER = "/tmp/extracted"
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    EXTRACT_FOLDER = os.path.join(os.path.dirname(__file__), "extracted")

# make sure the folders exist when the app starts
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# Initialize the database (this creates the tables if they don't exist built on the models)
init_db()


def is_supported_file(filename):
    """Check if a file has a supported extension for resume parsing."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALL_SUPPORTED_EXTENSIONS


def process_extracted_files(folder_path):
    """
    Walk through a folder and parse every supported resume file.
    Returns a list of parsed resume dicts.
    """
    parsed_resumes = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            # skip hidden files and macOS resource fork files
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
    Accepts resumes in two modes:
    1. ZIP mode (backward compatible): A single ZIP file containing PDF/DOCX/DOC/image resumes
    2. Multi-file mode: Multiple individual resume files (PDF, DOCX, DOC, PNG, JPG, etc.)
    
    Parses each resume, scores them, and returns the ranked results as JSON.
    """

    job_description = request.form.get("job_description", "")

    # Determine upload mode: ZIP file or multiple individual files
    has_zip = "file" in request.files and request.files["file"].filename.endswith(".zip")
    has_files = "files" in request.files

    if not has_zip and not has_files:
        # backward compat: check if 'file' has a non-zip resume file
        if "file" in request.files:
            single_file = request.files["file"]
            if single_file.filename and is_supported_file(single_file.filename):
                # treat single non-zip file as a one-file upload
                has_files = True
                request.files = {"files": single_file}
            else:
                return jsonify({"error": "No supported files uploaded. Please upload PDF, DOCX, DOC, or image files (individually or in a ZIP)."}), 400
        else:
            return jsonify({"error": "No file was uploaded."}), 400

    try:
        # clear out any old extracted files so we don't mix batches
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)

        parsed_resumes = []

        if has_zip:
            # === ZIP MODE ===
            file = request.files["file"]
            print(f"got a ZIP file: {file.filename}, starting to process...")

            # save the uploaded zip to our uploads folder
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)
            print(f"saved zip to {zip_path}")

            # simulate uploading to S3
            mock_upload_to_s3(zip_path)

            # unzip the file
            print("unzipping the file...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(EXTRACT_FOLDER)
            print("unzip done!")

            # parse all supported files from the extracted folder
            parsed_resumes = process_extracted_files(EXTRACT_FOLDER)

        else:
            # === MULTI-FILE MODE ===
            files = request.files.getlist("files")
            if not files:
                # fallback: single file sent with key 'files'
                single = request.files.get("files")
                files = [single] if single else []

            print(f"got {len(files)} individual file(s), starting to process...")

            for f in files:
                if not f or not f.filename:
                    continue
                if not is_supported_file(f.filename):
                    print(f"skipping unsupported file: {f.filename}")
                    continue

                # save each file to the extract folder
                safe_name = f"{uuid.uuid4().hex[:8]}_{f.filename}"
                filepath = os.path.join(EXTRACT_FOLDER, safe_name)
                f.save(filepath)
                print(f"saved: {f.filename} -> {safe_name}")

                # simulate S3 upload for each file
                mock_upload_to_s3(filepath)

                # parse the resume
                result = parse_resume(filepath)
                if result is not None:
                    # use original filename, not the uuid-prefixed one
                    result["filename"] = f.filename
                    parsed_resumes.append(result)

        if len(parsed_resumes) == 0:
            return jsonify({"error": "No valid resume files found. Supported formats: PDF, DOCX, DOC, PNG, JPG, JPEG, TIFF, BMP"}), 400

        print(f"parsed {len(parsed_resumes)} resumes, now scoring...")

        # score each parsed resume, passing the optional job_description
        scored_candidates = []
        for resume_data in parsed_resumes:
            scored = score_candidate(resume_data, job_description)
            scored_candidates.append(scored)

        # rank them by score (highest first) BEFORE anomaly detection
        # so JSON gets the full data including raw_text
        ranked_candidates = rank_candidates(scored_candidates)

        # Generate a unique batch_id for this upload
        batch_id = str(uuid.uuid4())

        # Save to JSON file FIRST (preserves raw_text for future ML model training)
        # detect_anomalies() below will strip raw_text, so we must save before that
        save_to_json(ranked_candidates, batch_id, job_description)

        # Now run anomaly detection (this strips raw_text from the dicts)
        ranked_candidates = detect_anomalies(ranked_candidates)

        # Save to the database (for dashboard display, without raw_text)
        save_candidates(ranked_candidates, batch_id)

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
    Returns the latest batch of processed results directly from the MySQL database.
    The dashboard page calls this when it loads.
    """
    # Fetch from database instead of the old in-memory variable
    latest_results = get_latest_batch()
    
    return jsonify({
        "candidates": latest_results,
        "count": len(latest_results)
    }), 200


@app.route("/json-data", methods=["GET"])
def get_json_data():
    """
    Returns the full JSON storage data (all batches).
    Useful for debugging and for future ML model training endpoints.
    """
    data = load_json_data()
    return jsonify(data), 200


if __name__ == "__main__":
    # using port 5001 because macOS uses 5000 for AirPlay
    print("starting HireFlow AI backend on http://localhost:5001")
    app.run(debug=True, port=5001)
