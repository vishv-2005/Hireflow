// LandingPage.jsx - the main landing page with hero section and features
// this is what visitors see first when they open the app

import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import FeatureCard from '../components/FeatureCard';
import { API_BASE_URL } from '../config';
import '../styles/landing.css';

function LandingPage() {
    const navigate = useNavigate();

    // the three features we want to highlight on the landing page
    const features = [
        {
            illustration: (
                <div className="ui-mockup mockup-parse">
                    <div className="mockup-file">
                        <div className="mockup-file-header">
                            <div className="mockup-point"></div>
                            <div className="mockup-point"></div>
                            <div className="mockup-point"></div>
                        </div>
                        <div className="mockup-line w-full"></div>
                        <div className="mockup-line w-3-4"></div>
                        <div className="mockup-line w-1-2"></div>
                    </div>
                    <div className="mockup-scanner"></div>
                </div>
            ),
            title: 'Multimodal Resume Parsing',
            description:
                'Upload PDFs, DOCX files, or ZIP archives. Our pipeline extracts and structures every piece of candidate data automatically.',
        },
        {
            illustration: (
                <div className="ui-mockup mockup-score">
                    <div className="mockup-candidate">
                        <div className="mockup-avatar"></div>
                        <div className="mockup-info">
                            <div className="mockup-name"></div>
                            <div className="mockup-role"></div>
                        </div>
                        <div className="mockup-badge">94.5</div>
                    </div>
                </div>
            ),
            title: 'Smart Candidate Scoring',
            description:
                'Each resume is scored against a keyword model trained on real job market data. No more manual shortlisting.',
        },
        {
            illustration: (
                <div className="ui-mockup mockup-anomaly">
                    <div className="mockup-bg-lines">
                        <div className="mockup-line w-full"></div>
                        <div className="mockup-line w-full op-20"></div>
                        <div className="mockup-line w-3-4 op-20"></div>
                    </div>
                    <div className="mockup-alert">
                        <div className="mockup-alert-dot"></div>
                        <div className="mockup-alert-text">Keyword stuffing detected</div>
                    </div>
                </div>
            ),
            title: 'Anomaly Detection',
            description:
                'We flag resumes with keyword stuffing and suspicious patterns so your HR team only sees genuine candidates.',
        },
    ];

    // state for real pipeline stats replacing hardcoded fake data
    const [stats, setStats] = useState({ total: '—', avgScore: '—', anomalies: '—' });
    const [loadingStats, setLoadingStats] = useState(true);

    // refs for scroll animations on the impact section
    const impactRef1 = useRef(null);
    const impactRef2 = useRef(null);
    const impactRef3 = useRef(null);
    const [visibleStats, setVisibleStats] = useState({ stat1: false, stat2: false, stat3: false });

    // fetch the latest batch results from the API when component mounts
    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await axios.get(`${API_BASE_URL}/results`);
                const candidates = response.data.candidates || [];

                if (candidates.length > 0) {
                    const total = candidates.length;
                    const avgScore = (candidates.reduce((sum, c) => sum + c.score, 0) / total).toFixed(1);
                    const anomalies = candidates.filter(c => c.is_anomaly).length;

                    setStats({ total, avgScore, anomalies });
                }
            } catch (err) {
                console.error('Failed to fetch stats for landing page:', err);
            } finally {
                setLoadingStats(false);
            }
        };

        fetchStats();
    }, []);

    // Intersection Observer to trigger scroll animations when impact section comes into view
    useEffect(() => {
        const observerOptions = {
            threshold: 0.3, // Trigger when 30% of the element is visible
            rootMargin: '0px 0px -50px 0px'
        };

        const observerCallback = (entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    if (entry.target === impactRef1.current) setVisibleStats(prev => ({ ...prev, stat1: true }));
                    if (entry.target === impactRef2.current) setVisibleStats(prev => ({ ...prev, stat2: true }));
                    if (entry.target === impactRef3.current) setVisibleStats(prev => ({ ...prev, stat3: true }));
                    observer.unobserve(entry.target);
                }
            });
        };

        const observer = new IntersectionObserver(observerCallback, observerOptions);

        if (impactRef1.current) observer.observe(impactRef1.current);
        if (impactRef2.current) observer.observe(impactRef2.current);
        if (impactRef3.current) observer.observe(impactRef3.current);

        return () => observer.disconnect();
    }, []);

    return (
        <div className="landing-page">
            {/* hero section - split layout with text on left and card on right */}
            <section className="hero">
                <div className="hero-left">
                    <span className="hero-badge">AI-Powered Recruitment Analytics</span>
                    <h1 className="hero-heading">
                        Find the right talent, faster.
                    </h1>
                    <p className="hero-subtext">
                        HireFlow AI is a Big Data analytics pipeline built for modern HR teams.
                        Upload resumes, score candidates, and make data-driven hiring decisions
                        — all in one place.
                    </p>
                    <div className="hero-buttons">
                        <button
                            className="btn-primary"
                            onClick={() => navigate('/upload')}
                        >
                            Get Started →
                        </button>
                        <button
                            className="btn-secondary"
                            onClick={() => navigate('/dashboard')}
                        >
                            View Demo
                        </button>
                    </div>
                </div>

                {/* floating preview card on the right side */}
                <div className="hero-right">
                    <div className="hero-preview-card">
                        <div className="preview-card-header">Live Pipeline Stats</div>
                        {/* using real data fetched from the backend so marketing site matches reality */}
                        <div className="preview-stat">
                            <span className="preview-stat-label">Candidates Ranked</span>
                            <span className="preview-stat-value highlight">
                                {loadingStats ? 'Loading...' : stats.total}
                            </span>
                        </div>
                        <div className="preview-stat">
                            <span className="preview-stat-label">Avg. Score</span>
                            <span className="preview-stat-value">
                                {loadingStats ? 'Loading...' : stats.avgScore}
                            </span>
                        </div>
                        <div className="preview-stat">
                            <span className="preview-stat-label">Anomalies Detected</span>
                            <span className="preview-stat-value">
                                {loadingStats ? 'Loading...' : stats.anomalies}
                            </span>
                        </div>
                    </div>
                    <p style={{ textAlign: 'center', color: 'var(--sage)', fontSize: '0.85rem', marginTop: '12px' }}>
                        Live data from last processed batch.
                    </p>
                </div>
            </section>

            {/* features section - three cards in a row */}
            <section className="features-section">
                <h2 className="features-section-title">How It Works</h2>
                <p className="features-section-subtitle">
                    From upload to insight — our pipeline handles it all.
                </p>
                <div className="features-grid">
                    {features.map((feature, index) => (
                        <FeatureCard
                            key={index}
                            illustration={feature.illustration}
                            title={feature.title}
                            description={feature.description}
                        />
                    ))}
                </div>
            </section>

            {/* impact section - inspired by premium data workflows, moved below features integration */}
            <section className="impact-section">
                <div className="impact-container">
                    <div className="impact-left">
                        <h2>Automate your most analytical<br />multi-step recruitment workflows</h2>
                    </div>
                    <div className="impact-right">
                        <div ref={impactRef1} className={`impact-stat ${visibleStats.stat1 ? 'is-visible' : ''}`}>
                            <span className="impact-number">2min</span>
                            <span className="impact-label">to score initial candidate pool</span>
                        </div>
                        <div ref={impactRef2} className={`impact-stat delay-1 ${visibleStats.stat2 ? 'is-visible' : ''}`}>
                            <span className="impact-number">8x</span>
                            <span className="impact-label">faster resume reviews</span>
                        </div>
                        <div ref={impactRef3} className={`impact-stat delay-2 ${visibleStats.stat3 ? 'is-visible' : ''}`}>
                            <span className="impact-number">50%</span>
                            <span className="impact-label">time reduction in hiring</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* premium footer */}
            <footer className="footer-premium">
                <div className="footer-container">
                    <div className="footer-brand">
                        <div className="navbar-logo">Hire<span>Flow</span></div>
                        <p>Built for HR teams that move fast.</p>
                    </div>
                    <div className="footer-links">
                        <div className="link-group">
                            <h4>Product</h4>
                            <a href="#">Features</a>
                            <a href="#">Demo</a>
                            <a href="#">Pricing</a>
                        </div>
                        <div className="link-group">
                            <h4>Company</h4>
                            <a href="#">About</a>
                            <a href="#">Blog</a>
                            <a href="#">Careers</a>
                        </div>
                        <div className="link-group">
                            <h4>Legal</h4>
                            <a href="#">Privacy</a>
                            <a href="#">Terms</a>
                        </div>
                    </div>
                </div>
                <div className="footer-bottom">
                    <p>HireFlow AI © 2026. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
}

export default LandingPage;
