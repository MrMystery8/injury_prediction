import React, { useState } from 'react';
import axios from 'axios';
import { Settings, Play, Activity } from 'lucide-react';

function TrainingLab() {
    const [params, setParams] = useState({
        learningRate: 0.05,
        maxIter: 500,
        maxDepth: 6
    });
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

    const trainModel = () => {
        setLoading(true);
        axios.post('http://localhost:8000/api/run_step', {
            step_id: 'step_3_train',
            params: params
        })
            .then(res => {
                setResult({
                    ...res.data,
                    stdout: (res.data.stdout || '') + (res.data.stderr ? '\n=== LOGS (STDERR) ===\n' + res.data.stderr : '')
                });
            })
            .finally(() => setLoading(false));
    };

    return (
        <div>
            <div className="header-container">
                <h1 className="page-title">Training Lab</h1>
                <p className="page-desc">Experiment with hyperparameters and re-train the model.</p>
            </div>

            <div className="grid-cols-3">
                {/* Controls */}
                <div className="card" style={{ height: 'fit-content' }}>
                    <h2 className="section-title">
                        <Settings size={20} className="text-gray" />
                        Hyperparameters
                    </h2>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                            <label className="label">Learning Rate</label>
                            <input
                                type="number"
                                step="0.01"
                                value={params.learningRate}
                                onChange={e => setParams({ ...params, learningRate: parseFloat(e.target.value) })}
                                className="input-field"
                            />
                        </div>
                        <div>
                            <label className="label">Max Iterations (Trees)</label>
                            <input
                                type="number"
                                value={params.maxIter}
                                onChange={e => setParams({ ...params, maxIter: parseInt(e.target.value) })}
                                className="input-field"
                            />
                        </div>
                        <div>
                            <label className="label">Max Depth</label>
                            <input
                                type="number"
                                value={params.maxDepth}
                                onChange={e => setParams({ ...params, maxDepth: parseInt(e.target.value) })}
                                className="input-field"
                            />
                        </div>

                        <button
                            onClick={trainModel}
                            disabled={loading}
                            className="btn btn-primary"
                            style={{ width: '100%', justifyContent: 'center', marginTop: '1rem' }}
                        >
                            {loading ? 'Training...' : <><Play size={18} /> Start Training</>}
                        </button>
                    </div>
                </div>

                {/* Results */}
                <div style={{ gridColumn: 'span 2' }}>
                    {result ? (
                        <div className="bg-dark rounded-lg p-4" style={{ fontFamily: 'monospace', fontSize: '0.875rem', overflow: 'auto', maxHeight: '600px', whiteSpace: 'pre-wrap' }}>
                            {result.stdout}
                            {result.stderr && <div className="text-red mt-4 pt-4 border-t border-gray-700">{result.stderr}</div>}
                        </div>
                    ) : (
                        <div
                            style={{
                                height: '300px',
                                border: '2px dashed var(--border)',
                                borderRadius: 'var(--radius-lg)',
                                backgroundColor: '#f8fafc',
                                display: 'flex',
                                flexDirection: 'column',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'var(--text-secondary)'
                            }}
                        >
                            <Activity size={48} style={{ opacity: 0.2, marginBottom: '1rem' }} />
                            <p style={{ fontWeight: 500 }}>Ready to Train</p>
                            <span className="text-sm">Adjust parameters and click Start</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default TrainingLab;
