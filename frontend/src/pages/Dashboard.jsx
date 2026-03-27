// Dashboard.jsx - shows the results of processed resumes
// fetches data from Flask backend on load and displays stats + candidate table

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { API_BASE_URL } from '../config';
import CandidateTable from '../components/CandidateTable';
import '../styles/dashboard.css';

function Dashboard() {
    const navigate = useNavigate();

    // state for the candidates data from the backend
    const [candidates, setCandidates] = useState([]);
    const [loading, setLoading] = useState(true);

    // frontend filter/search states
    const [searchQuery, setSearchQuery] = useState('');
    const [showFlaggedOnly, setShowFlaggedOnly] = useState(false);
    const [topN, setTopN] = useState('all'); // controls how many top candidates to show

    // fetch results from Flask when the component mounts
    useEffect(() => {
        const fetchResults = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/results`);
                console.log('got results:', response.data);
                setCandidates(response.data.candidates || []);
            } catch (err) {
                console.error('failed to fetch results:', err);
                // if the backend isn't running or has no data, just show empty state
                setCandidates([]);
            } finally {
                setLoading(false);
            }
        };

        fetchResults();
    }, []);

    // calculate stats for the top bar
    const totalResumes = candidates.length;
    const avgScore =
        totalResumes > 0
            ? (candidates.reduce((sum, c) => sum + c.score, 0) / totalResumes).toFixed(1)
            : '0.0';
    const topCandidate = totalResumes > 0 ? candidates[0]?.name : '—';

    // apply our frontend filters to the loaded candidate list
    const filteredCandidates = candidates
        .filter((candidate) => {
            const searchLower = searchQuery.toLowerCase();
            const matchesSearch =
                candidate.name.toLowerCase().includes(searchLower) ||
                candidate.matched_skills.some(skill => skill.toLowerCase().includes(searchLower));
            const matchesFlagged = showFlaggedOnly ? candidate.is_anomaly === true : true;
            return matchesSearch && matchesFlagged;
        })
        // apply top-n slice. candidates are already sorted by score desc from the backend
        .slice(0, topN === 'all' ? undefined : parseInt(topN, 10));

    // loading state while we wait for the backend response
    if (loading) {
        return (
            <div className="dashboard-page">
                <div className="upload-loading" style={{ marginTop: '80px' }}>
                    <div className="upload-spinner"></div>
                    <p className="upload-loading-text">Loading results...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-page">
            {/* header with title and upload more button */}
            <div className="dashboard-header">
                <h1 className="dashboard-title">Dashboard</h1>
                <button
                    className="btn-primary"
                    onClick={() => navigate('/upload')}
                >
                    Upload More
                </button>
            </div>

            {candidates.length > 0 ? (
                <>
                    {/* stats bar at the top - three boxes, no emojis */}
                    <div className="stats-bar">
                        <div className="stat-box">
                            <div className="stat-icon-bar stat-icon-blue"></div>
                            <div className="stat-value">{totalResumes}</div>
                            <div className="stat-label">Total Resumes</div>
                        </div>
                        <div className="stat-box">
                            <div className="stat-icon-bar stat-icon-orange"></div>
                            <div className="stat-value">{avgScore}</div>
                            <div className="stat-label">Average Score</div>
                        </div>
                        <div className="stat-box">
                            <div className="stat-icon-bar stat-icon-sage"></div>
                            <div className="stat-value stat-value-sm">{topCandidate}</div>
                            <div className="stat-label">Top Candidate</div>
                        </div>
                    </div>

                    {/* search, filter, and top-n control bar above the table */}
                    <div className="dashboard-controls">
                        <input
                            type="text"
                            placeholder="Search by name or skill..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="search-input"
                        />
                        <button
                            onClick={() => setShowFlaggedOnly(!showFlaggedOnly)}
                            className={`flagged-btn ${showFlaggedOnly ? 'active' : ''}`}
                        >
                            {showFlaggedOnly ? 'Showing Flagged' : 'Show Flagged Only'}
                        </button>
                        <div className="topn-control">
                            <label htmlFor="topn-select" className="topn-label">View:</label>
                            <select
                                id="topn-select"
                                value={topN}
                                onChange={(e) => setTopN(e.target.value)}
                                className="topn-select"
                            >
                                <option value="5">Top 5</option>
                                <option value="10">Top 10</option>
                                <option value="20">Top 20</option>
                                <option value="50">Top 50</option>
                                <option value="all">All</option>
                            </select>
                        </div>
                    </div>

                    {/* determine whether to show the table or a generic 'no match' message */}
                    {filteredCandidates.length > 0 ? (
                        <CandidateTable candidates={filteredCandidates} />
                    ) : (
                        <div style={{ textAlign: 'center', padding: '40px', backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #eaeaea' }}>
                            <p style={{ color: '#666', fontSize: '1.05rem', margin: 0 }}>
                                No candidates match your filters. Try adjusting your search.
                            </p>
                        </div>
                    )}
                </>
            ) : (
                // empty state - no results yet
                <div className="dashboard-empty">
                    <div className="dashboard-empty-visual"></div>
                    <h2 className="dashboard-empty-title">No Results Yet</h2>
                    <p className="dashboard-empty-text">
                        Upload a ZIP file of resumes to see candidate rankings here.
                    </p>
                    <button
                        className="btn-primary"
                        onClick={() => navigate('/upload')}
                    >
                        Upload Resumes
                    </button>
                </div>
            )}
        </div>
    );
}

export default Dashboard;
