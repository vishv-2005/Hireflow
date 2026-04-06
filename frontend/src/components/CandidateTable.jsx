// CandidateTable.jsx - displays the ranked candidates in a table
// scores get colored badges: green for 70+, orange for 40-69, red for below 40
// skills that match the JD are highlighted with a distinct color

import '../styles/dashboard.css';

function CandidateTable({ candidates }) {
    // this function figures out what color class to use for the score badge
    const getScoreClass = (score) => {
        if (score >= 70) return 'high';
        if (score >= 40) return 'medium';
        return 'low';
    };

    // count how many candidates were flagged so we can show a summary banner
    const flaggedCount = candidates.filter(c => c.is_anomaly).length;

    // generates a CSV blob from the currently visible candidates and triggers a download
    // using pure vanilla JS!
    const handleExportCSV = () => {
        // defined CSV headers
        const headers = ['Rank', 'Name', 'Score', 'Experience (Yrs)', 'Relevant Cert', 'Relevant Projects', 'Matched Skills', 'JD Matched Skills', 'Anomaly Status', 'Filename'];

        // build rows
        const rows = candidates.map(c => [
            c.rank,
            `"${c.name.replace(/"/g, '""')}"`, // escape quotes
            c.score,
            c.experience_years || 0,
            c.has_relevant_cert ? 'Yes' : 'No',
            c.relevant_projects_count || 0,
            `"${c.matched_skills.join(', ')}"`, // wrap list in quotes so commas don't break columns
            `"${(c.jd_matched_skills || []).join(', ')}"`,
            c.is_anomaly ? 'Flagged' : 'Clean',
            `"${c.filename}"`
        ]);

        // join headers and rows into one big string with newlines
        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        // create a Blob from the string so the browser treats it as a file
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);

        // create a hidden anchor link, click it, and clean up
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'hireflow-results.csv');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="results-section">
            <div className="results-section-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <h3 className="results-section-title" style={{ display: 'inline-block', marginRight: '16px' }}>Candidate Rankings</h3>
                    <span style={{ color: '#888', fontSize: '0.9rem' }}>
                        {candidates.length} candidate{candidates.length !== 1 ? 's' : ''}
                    </span>
                </div>
                <button
                    onClick={handleExportCSV}
                    style={{
                        padding: '6px 12px', fontSize: '0.85rem', fontWeight: 600,
                        backgroundColor: '#fff', border: '1px solid #ddd',
                        borderRadius: '6px', cursor: 'pointer', color: '#333'
                    }}
                >
                    ⬇ Export CSV
                </button>
            </div>

            {/* anomaly summary banner - only shows if there's at least one flagged candidate */}
            {flaggedCount > 0 && (
                <div style={{ backgroundColor: '#fef3c7', color: '#92400e', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px', fontSize: '0.95rem', display: 'flex', alignItems: 'center' }}>
                    <span style={{ marginRight: '8px' }}>⚠️</span>
                    <strong>{flaggedCount} of {candidates.length} candidate{candidates.length !== 1 ? 's' : ''} flagged for anomalies.</strong>
                </div>
            )}

            <table className="candidate-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Candidate Name</th>
                        <th>Score</th>
                        <th>Experience</th>
                        <th>Certificates</th>
                        <th>Projects</th>
                        <th>Status</th>
                        <th>Matched Skills</th>
                        <th>File Name</th>
                    </tr>
                </thead>
                <tbody>
                    {candidates.map((candidate) => {
                        // Build a set for quick lookup
                        const jdSkillSet = new Set((candidate.jd_matched_skills || []).map(s => s.toLowerCase()));
                        const projCount = candidate.relevant_projects_count || 0;
                        const projScore = candidate.project_relevance_score || 0;

                        return (
                            <tr key={candidate.rank}>
                                <td>
                                    <span className={`rank-badge ${candidate.rank <= 3 ? 'top-3' : ''}`}>
                                        {candidate.rank}
                                    </span>
                                </td>
                                <td className="candidate-name">
                                    {candidate.name}
                                </td>
                                <td>
                                    {/* add a red border if this candidate is an anomaly to reinforce the warning visually */}
                                    <span
                                        className={`score-badge ${getScoreClass(candidate.score)}`}
                                        style={candidate.is_anomaly ? { border: '2px solid #e53e3e' } : {}}
                                    >
                                        {candidate.score}%
                                    </span>
                                </td>
                                <td>
                                    <span style={{ fontWeight: 600, color: '#333' }}>
                                        {candidate.experience_years || 0} yrs
                                    </span>
                                </td>
                                <td>
                                    {candidate.has_relevant_cert ? (
                                        <span style={{ color: '#2f855a', fontWeight: '600' }}>✓ Yes</span>
                                    ) : (
                                        <span style={{ color: '#888' }}>-</span>
                                    )}
                                </td>
                                <td>
                                    {projCount > 0 ? (
                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '2px' }}>
                                            <span className="project-badge relevant">
                                                {projCount} relevant
                                            </span>
                                            <span style={{ fontSize: '0.72rem', color: '#888' }}>
                                                {(projScore * 100).toFixed(0)}% match
                                            </span>
                                        </div>
                                    ) : (
                                        <span style={{ color: '#888' }}>–</span>
                                    )}
                                </td>
                                <td>
                                    {/* Status column shows a red badge for anomalies or a green badge if clean */}
                                    {candidate.is_anomaly ? (
                                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                                            <span style={{ backgroundColor: '#fed7d7', color: '#c53030', padding: '4px 8px', borderRadius: '12px', fontSize: '0.85rem', fontWeight: 600 }}>
                                                ⚠ Flagged
                                            </span>
                                            <span title={candidate.anomaly_reason} style={{ fontSize: '0.75rem', color: '#e53e3e', marginTop: '4px', cursor: 'help', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '120px' }}>
                                                {candidate.anomaly_reason}
                                            </span>
                                        </div>
                                    ) : (
                                        <span style={{ backgroundColor: '#c6f6d5', color: '#2f855a', padding: '4px 8px', borderRadius: '12px', fontSize: '0.85rem', fontWeight: 600 }}>
                                            ✓ Clean
                                        </span>
                                    )}
                                </td>
                                <td>
                                    <div className="skills-list">
                                        {candidate.matched_skills.length > 0 ? (
                                            <>
                                                {/* Show JD-matched skills first, then the rest */}
                                                {candidate.matched_skills
                                                    .slice()
                                                    .sort((a, b) => {
                                                        const aMatch = jdSkillSet.has(a.toLowerCase()) ? 0 : 1;
                                                        const bMatch = jdSkillSet.has(b.toLowerCase()) ? 0 : 1;
                                                        return aMatch - bMatch;
                                                    })
                                                    .map((skill, index) => {
                                                        const isJdMatch = jdSkillSet.has(skill.toLowerCase());
                                                        return (
                                                            <span
                                                                className={`skill-pill ${isJdMatch ? 'skill-pill-jd' : ''}`}
                                                                key={index}
                                                                title={isJdMatch ? '✓ Matches Job Description' : ''}
                                                            >
                                                                {isJdMatch && <span className="skill-pill-jd-dot"></span>}
                                                                {skill}
                                                            </span>
                                                        );
                                                    })}
                                            </>
                                        ) : (
                                            <span style={{ color: '#999', fontSize: '0.85rem' }}>
                                                No matches
                                            </span>
                                        )}
                                    </div>
                                </td>
                                <td className="file-name">{candidate.filename}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

export default CandidateTable;
