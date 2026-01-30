import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Database, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

function DataView({ datasetId, label }) {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const loadData = () => {
        setLoading(true);
        axios.get(`${API_BASE}/data/${datasetId}?limit=100`)
            .then(res => {
                setData(res.data);
                setError(null);
            })
            .catch(err => {
                console.error(err);
                setError("Failed to load data. Is the backend running?");
            })
            .finally(() => setLoading(false));
    };

    useEffect(() => {
        if (datasetId) loadData();
    }, [datasetId]);

    if (loading) return <div className="p-4 text-gray">Loading {label}...</div>;
    if (error) return <div className="p-4 text-red">{error}</div>;
    if (!data) return null;

    return (
        <div style={{ marginTop: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', overflow: 'hidden', backgroundColor: 'white' }}>
            <div style={{ padding: '1rem', borderBottom: '1px solid var(--border)', backgroundColor: '#f8fafc', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h3 style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{label} <span style={{ fontWeight: 400, fontSize: '0.85em', color: 'var(--text-secondary)' }}>(Preview 100 rows)</span></h3>
                <button onClick={loadData} className="btn btn-ghost">
                    <RefreshCw size={16} />
                </button>
            </div>
            <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '0.875rem', textAlign: 'left', borderCollapse: 'collapse' }}>
                    <thead style={{ backgroundColor: '#f8fafc', color: 'var(--text-secondary)', fontWeight: 500 }}>
                        <tr>
                            {data.columns.map(col => (
                                <th key={col.name} style={{ padding: '0.75rem 1rem', whiteSpace: 'nowrap' }}>
                                    {col.name} <span style={{ fontSize: '0.75em', fontWeight: 400, color: '#94a3b8' }}>({col.type})</span>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody style={{ divideY: '1px solid var(--border)' }}>
                        {data.data.map((row, i) => (
                            <tr key={i} style={{ borderTop: '1px solid var(--border)', transition: 'background-color 0.1s' }} onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f8fafc'} onMouseLeave={e => e.currentTarget.style.backgroundColor = 'transparent'}>
                                {data.columns.map(col => (
                                    <td key={col.name} style={{ padding: '0.75rem 1rem', whiteSpace: 'nowrap', color: '#334155' }}>
                                        {row[col.name]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}

function DataExplorer() {
    const [selectedId, setSelectedId] = useState('clean_panel');

    const datasets = [
        { id: 'raw_injuries', label: 'Raw Injuries (CSV)' },
        { id: 'raw_transfers', label: 'Raw Transfers (CSV)' },
        { id: 'raw_profiles', label: 'Raw Profiles (CSV)' },
        { id: 'clean_panel', label: 'Process Panel (Parquet)' }
    ];

    return (
        <div>
            <div className="header-container">
                <h1 className="page-title">Data Explorer</h1>
                <p className="page-desc">Inspect raw sources and processed tables directly from the pipeline.</p>
            </div>

            <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                {datasets.map(ds => (
                    <button
                        key={ds.id}
                        onClick={() => setSelectedId(ds.id)}
                        className={`btn ${selectedId === ds.id ? 'btn-primary' : 'btn-secondary'}`}
                    >
                        <Database size={16} />
                        {ds.label}
                    </button>
                ))}
            </div>

            <DataView datasetId={selectedId} label={datasets.find(d => d.id === selectedId)?.label} />
        </div>
    );
}

export default DataExplorer;
