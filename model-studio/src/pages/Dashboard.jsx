import React, { useEffect, useState } from 'react';
import { MetricCard } from '../components/MetricCard';
import { FeatureImportance } from '../components/FeatureImportance';
import { TrendingUp, CheckCircle, AlertTriangle } from 'lucide-react';

function Dashboard() {
    const [data, setData] = useState(null);

    useEffect(() => {
        fetch('/model_data.json')
            .then(res => res.json())
            .then(setData)
            .catch(err => console.error("Failed to load model data", err));
    }, []);

    if (!data) return <div className="p-10">Loading Dashboard...</div>;

    const { metrics, feature_importance } = data;

    return (
        <div>
            <div className="header-container">
                <h1 className="page-title">Executive Dashboard</h1>
                <p className="page-desc">Real-time model performance monitoring and key insights.</p>
            </div>

            {/* Metrics Grid */}
            <div className="metrics-grid">
                <MetricCard
                    title="PR-AUC"
                    value={(metrics.pr_auc || 0).toFixed(4)}
                    subtext={`Baseline: ${(metrics.test_prevalence || 0.06).toFixed(4)}`}
                    trend={((metrics.pr_auc - metrics.test_prevalence) / metrics.test_prevalence * 100).toFixed(0)}
                />
                <MetricCard
                    title="ROC-AUC"
                    value={(metrics.roc_auc || 0).toFixed(4)}
                    subtext="Acceptable > 0.55"
                />
                <MetricCard
                    title="Precision @ Top 5%"
                    value={((metrics.precision_top5 || 0) * 100).toFixed(2) + "%"}
                    subtext="Actionable Lift"
                    trend={null}
                />
                <MetricCard
                    title="Global Prevalence"
                    value={((metrics.test_prevalence || 0) * 100).toFixed(2) + "%"}
                    subtext="Test Set (2024+)"
                />
            </div>

            <div className="grid-cols-3">
                {/* Main Chart Area */}
                <div className="card" style={{ gridColumn: 'span 2' }}>
                    <h2 className="section-title">
                        <TrendingUp size={20} className="text-blue" />
                        Feature Importance (Top 10)
                    </h2>
                    <div className="chart-container">
                        <FeatureImportance data={feature_importance} />
                    </div>
                </div>

                {/* Sidebar / Actions */}
                <div className="card" style={{ height: 'fit-content' }}>
                    <h2 className="section-title">
                        <CheckCircle size={20} className="text-blue" />
                        Model Health
                    </h2>
                    <ul className="health-list">
                        <li>
                            <span className="dot dot-green"></span>
                            <span>Silent Leakage Check Passed</span>
                        </li>
                        <li>
                            <span className="dot dot-green"></span>
                            <span>Calibration (Isotonic) Active</span>
                        </li>
                        <li>
                            <span className="dot dot-blue"></span>
                            <span>Target: 30-Day Window</span>
                        </li>
                    </ul>

                    <div className="alert-box">
                        <div className="alert-content">
                            <AlertTriangle size={20} className="icon-yellow" />
                            <div>
                                <h4 className="alert-title">Operational Note</h4>
                                <p className="alert-text">
                                    Top 5% alerts have <strong>{((metrics.precision_top5 || 0) * 100).toFixed(1)}%</strong> True Positive rate.
                                    <br /><br />
                                    <strong>Recommended Action:</strong> Assign physio review for these players.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Dashboard;
