// UploadPage.jsx - where users upload their resumes
// handles drag and drop, file selection (ZIP or individual files), upload to Flask, and shows results

import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import '../styles/upload.css';

// all supported resume file extensions
const SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'];
const ACCEPT_STRING = '.zip,.pdf,.docx,.doc,.png,.jpg,.jpeg,.tiff,.bmp,.webp';

function isSupported(filename) {
    const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
    return ext === '.zip' || SUPPORTED_EXTENSIONS.includes(ext);
}

function UploadPage() {
    const navigate = useNavigate();
    const fileInputRef = useRef(null);

    // state for tracking the upload flow
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadResult, setUploadResult] = useState(null);
    const [error, setError] = useState(null);
    const [isDragOver, setIsDragOver] = useState(false);
    const [jobDescription, setJobDescription] = useState('');

    // adds files to the selected list, validating extensions
    const addFiles = (fileList) => {
        const newFiles = Array.from(fileList);
        const valid = [];
        const invalid = [];

        for (const f of newFiles) {
            if (isSupported(f.name)) {
                valid.push(f);
            } else {
                invalid.push(f.name);
            }
        }

        if (invalid.length > 0) {
            setError(`Unsupported file(s): ${invalid.join(', ')}. Supported: PDF, DOCX, DOC, PNG, JPG, TIFF, BMP, or a ZIP.`);
        } else {
            setError(null);
        }

        if (valid.length > 0) {
            // if user picks a ZIP, use only the ZIP (replace everything)
            const hasZip = valid.some(f => f.name.toLowerCase().endsWith('.zip'));
            if (hasZip) {
                // only keep the first ZIP
                const zip = valid.find(f => f.name.toLowerCase().endsWith('.zip'));
                setSelectedFiles([zip]);
            } else {
                // append to existing selection (dedup by name)
                setSelectedFiles(prev => {
                    const existingNames = new Set(prev.map(f => f.name));
                    const unique = valid.filter(f => !existingNames.has(f.name));
                    return [...prev, ...unique];
                });
            }
        }
    };

    // this handles when a user drops files onto the dropzone
    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragOver(false);
        setError(null);
        addFiles(e.dataTransfer.files);
    };

    // prevents the browser from opening the file when dragged over
    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = () => {
        setIsDragOver(false);
    };

    // when the user clicks the dropzone, open the file picker
    const handleDropzoneClick = () => {
        fileInputRef.current.click();
    };

    // handles file selection from the file picker dialog
    const handleFileChange = (e) => {
        setError(null);
        if (e.target.files.length > 0) {
            addFiles(e.target.files);
        }
        // reset so the same file can be selected again
        e.target.value = '';
    };

    // removes a single file from selection
    const handleRemoveFile = (index) => {
        setSelectedFiles(prev => prev.filter((_, i) => i !== index));
        setError(null);
    };

    // clears all selected files
    const handleClearAll = () => {
        setSelectedFiles([]);
        setError(null);
    };

    // sends the files to our Flask backend for processing
    const handleUpload = async () => {
        if (selectedFiles.length === 0) return;

        setIsUploading(true);
        setError(null);

        // we use FormData because that's how you send files with axios
        const formData = new FormData();

        const isZipMode = selectedFiles.length === 1 && selectedFiles[0].name.toLowerCase().endsWith('.zip');

        if (isZipMode) {
            // backward-compatible: send single ZIP with 'file' key
            formData.append('file', selectedFiles[0]);
        } else {
            // multi-file mode: send all files with 'files' key
            for (const f of selectedFiles) {
                formData.append('files', f);
            }
        }

        if (jobDescription.trim()) {
            formData.append('job_description', jobDescription.trim());
        }

        try {
            // posting to Flask backend using our dynamic configuration API_BASE_URL
            const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            console.log('upload response:', response.data);
            setUploadResult(response.data);
        } catch (err) {
            console.error('upload failed:', err);
            const errorData = err.response?.data?.error;
            const errorMsg = typeof errorData === 'string'
                ? errorData
                : (errorData?.message || 'Something went wrong. Is the backend running?');
            setError(errorMsg);
        } finally {
            setIsUploading(false);
        }
    };

    // helper to get icon for file type
    const getFileIcon = (filename) => {
        const ext = filename.toLowerCase().slice(filename.lastIndexOf('.'));
        if (ext === '.zip') return '📦';
        if (ext === '.pdf') return '📄';
        if (ext === '.docx' || ext === '.doc') return '📝';
        if (['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'].includes(ext)) return '🖼️';
        return '📎';
    };

    return (
        <div className="upload-page">
            <div className="upload-container">
                <h1 className="upload-title">Upload Resumes</h1>
                <p className="upload-subtitle">
                    Drop a ZIP or select individual resume files — we support
                    <strong> PDF, DOCX, DOC,</strong> and <strong>images</strong> (PNG, JPG, TIFF, BMP).
                </p>

                {/* show different content based on the current state */}
                {uploadResult ? (
                    // success state - upload is done
                    <div className="upload-success">
                        <span className="upload-success-icon">✅</span>
                        <h2 className="upload-success-title">Upload Complete!</h2>
                        <p className="upload-success-text">
                            {uploadResult.count} resume{uploadResult.count !== 1 ? 's' : ''} processed
                            successfully.
                        </p>
                        <button
                            className="btn-primary"
                            onClick={() => navigate('/dashboard')}
                        >
                            View Results on Dashboard
                        </button>
                    </div>
                ) : isUploading ? (
                    // loading state - files are being processed
                    <div className="upload-loading">
                        <div className="upload-spinner"></div>
                        <p className="upload-loading-text">Processing {selectedFiles.length} resume{selectedFiles.length !== 1 ? 's' : ''}...</p>
                        <p className="upload-loading-subtext">Extracting text, running OCR on images, and scoring candidates</p>
                    </div>
                ) : (
                    // default state - waiting for files
                    <>
                        {error && <div className="upload-error">⚠️ {error}</div>}

                        {/* the drag and drop zone */}
                        <div
                            className={`upload-dropzone ${isDragOver ? 'drag-over' : ''}`}
                            onDrop={handleDrop}
                            onDragOver={handleDragOver}
                            onDragLeave={handleDragLeave}
                            onClick={handleDropzoneClick}
                        >
                            <span className="upload-dropzone-icon">📁</span>
                            <p className="upload-dropzone-text">
                                Drop your files here or <strong>click to browse</strong>
                            </p>
                            <p className="upload-dropzone-hint">
                                ZIP, PDF, DOCX, DOC, PNG, JPG, TIFF, BMP
                            </p>
                        </div>

                        {/* hidden file input that gets triggered by clicking the dropzone */}
                        <input
                            type="file"
                            ref={fileInputRef}
                            className="upload-file-input"
                            accept={ACCEPT_STRING}
                            onChange={handleFileChange}
                            multiple
                        />

                        {/* show list of selected files */}
                        {selectedFiles.length > 0 && (
                            <div className="upload-file-list">
                                <div className="upload-file-list-header">
                                    <span className="upload-file-count">
                                        {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected
                                    </span>
                                    {selectedFiles.length > 1 && (
                                        <button className="upload-clear-all" onClick={handleClearAll}>
                                            Clear all
                                        </button>
                                    )}
                                </div>
                                {selectedFiles.map((file, index) => (
                                    <div key={`${file.name}-${index}`} className="upload-file-item">
                                        <span className="upload-file-name">
                                            {getFileIcon(file.name)} {file.name}
                                        </span>
                                        <span className="upload-file-size">
                                            {(file.size / 1024).toFixed(0)} KB
                                        </span>
                                        <button
                                            className="upload-file-remove"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleRemoveFile(index);
                                            }}
                                        >
                                            ✕
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* optional job description input */}
                        <div style={{ textAlign: 'left', marginBottom: '24px' }}>
                            <label style={{ display: 'block', fontSize: '0.95rem', fontWeight: 600, color: 'var(--charcoal)', marginBottom: '8px' }}>
                                Paste a job description to enable AI-powered matching (optional)
                            </label>
                            <textarea
                                value={jobDescription}
                                onChange={(e) => setJobDescription(e.target.value)}
                                placeholder="Paste the job requirements here... If left blank, we'll fall back to basic keyword matching."
                                style={{
                                    width: '100%', minHeight: '100px', padding: '12px',
                                    borderRadius: '8px', border: '1px solid var(--charcoal)',
                                    backgroundColor: 'var(--charcoal)', color: 'var(--white)',
                                    fontFamily: 'inherit', fontSize: '0.95rem', resize: 'vertical'
                                }}
                            />
                        </div>

                        {/* upload button - disabled until at least one file is selected */}
                        <button
                            className={`btn-primary upload-btn ${selectedFiles.length === 0 ? 'disabled' : ''}`}
                            onClick={handleUpload}
                            disabled={selectedFiles.length === 0}
                        >
                            Upload & Analyze {selectedFiles.length > 0 ? `(${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''})` : ''}
                        </button>
                    </>
                )}
            </div>
        </div>
    );
}

export default UploadPage;
