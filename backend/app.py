# app.py - main Flask backend for HireFlow AI
# this handles file uploads, resume parsing, and returning ranked results

import os
import zipfile
import shutil
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from resume_parser import parse_resume
from candidate_scorer import score_candidate, rank_candidates, detect_anomalies
from mock_s3 import mock_upload_to_s3
from database import init_db, save_candidates, get_latest_batch

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


@app.route("/upload", methods=["POST"])
def upload_zip():
    """
    Accepts a ZIP file from the frontend, unzips it,
    parses each resume inside, scores them, and returns
    the ranked results as JSON.
    """

    # check if a file was actually sent
    if "file" not in request.files:
        return jsonify({"error": "no file was uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "empty filename, please select a file"}), 400

    job_description = request.form.get("job_description", "")

    # make sure it's a zip file
    if not file.filename.endswith(".zip"):
        return jsonify({"error": "only ZIP files are accepted"}), 400

    try:
        print(f"got a file: {file.filename}, starting to process...")

        # save the uploaded zip to our uploads folder
        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(zip_path)
        print(f"saved zip to {zip_path}")

        # simulate uploading to S3 (this is just a print statement for now)
        mock_upload_to_s3(zip_path)

        # clear out any old extracted files so we don't mix batches
        if os.path.exists(EXTRACT_FOLDER):
            shutil.rmtree(EXTRACT_FOLDER)
        os.makedirs(EXTRACT_FOLDER, exist_ok=True)

        # unzip the file
        print("unzipping the file...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
        print("unzip done!")

        # walk through extracted files and parse each resume
        parsed_resumes = []
        for root, dirs, files in os.walk(EXTRACT_FOLDER):
            for fname in files:
                # skip hidden files and macOS resource fork files
                if fname.startswith(".") or fname.startswith("__"):
                    continue
                filepath = os.path.join(root, fname)
                print(f"parsing: {fname}")
                result = parse_resume(filepath)
                if result is not None:
                    parsed_resumes.append(result)

        if len(parsed_resumes) == 0:
            return jsonify({"error": "no valid resume files found in the ZIP"}), 400

        print(f"parsed {len(parsed_resumes)} resumes, now scoring...")

        # score each parsed resume, passing the optional job_description
        scored_candidates = []
        for resume_data in parsed_resumes:
            scored = score_candidate(resume_data, job_description)
            scored_candidates.append(scored)

        # run batch anomaly detection for keyword stuffing
        scored_candidates = detect_anomalies(scored_candidates)

        # rank them by score (highest first)
        ranked_candidates = rank_candidates(scored_candidates)

        # Generate a unique batch_id for this upload and save to the MySQL Database
        batch_id = str(uuid.uuid4())
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


if __name__ == "__main__":
    # using port 5001 because macOS uses 5000 for AirPlay
    print("starting HireFlow AI backend on http://localhost:5001")
    app.run(debug=True, port=5001)
