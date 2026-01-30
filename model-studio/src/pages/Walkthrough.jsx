import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import { Play, CheckCircle, Circle, Terminal as TerminalIcon, BookOpen } from 'lucide-react';
import 'xterm/css/xterm.css';

const STEPS = [
    // --- Phase 1 ---
    {
        id: 'step_1_1',
        title: '1.1 Detect Tables',
        desc: 'Scan raw folder for recognizable CSV signatures.',
        file: 'interactive_notebook.py',
        content: (
            <div className="space-y-4">
                <p>Instead of hardcoding filenames (which change between datasets), we scan headers.</p>
                <pre className="bg-gray-100 p-2 rounded text-xs font-mono">
                    {`signatures = {
  'injuries': {'player_id', 'injury_start', 'days_missed'},
  'market_val': {'player_id', 'date', 'market_value'}
}`}
                </pre>
            </div>
        )
    },
    {
        id: 'step_1_2',
        title: '1.2 Load Backbone',
        desc: 'Load Games, Appearances, Players, Competitions.',
        file: 'interactive_notebook.py',
        content: <p>We load the core relational schema. This data forms the "Backbone" of our study universe.</p>
    },
    {
        id: 'step_1_3',
        title: '1.3 Validate Backbone',
        desc: 'Check referential integrity and date formats.',
        file: 'interactive_notebook.py',
        content: <p>We verify that <code>player_id</code> is unique in the players table and that dates parse correctly.</p>
    },
    {
        id: 'step_1_4',
        title: '1.4 Load Enrichment',
        desc: 'Load Injuries, Transfers, Market Values.',
        file: 'interactive_notebook.py',
        content: <p>We ingest the auxiliary tables found in Step 1.1.</p>
    },

    // --- Phase 2 ---
    {
        id: 'step_2_1',
        title: '2.1 Clean Injuries',
        desc: 'Impute missing end dates and deduplicate.',
        file: 'interactive_notebook.py',
        content: (
            <div className="space-y-4">
                <p>Raw injury data often has missing <code>end_date</code>. We fix this:</p>
                <pre className="bg-gray-100 p-2 rounded text-xs font-mono">
                    {`# If end is missing, use start + days_out
df['end'] = df['start'] + pd.to_timedelta(df['days'], unit='D')`}
                </pre>
            </div>
        )
    },
    {
        id: 'step_2_2',
        title: '2.2 Clean Context',
        desc: 'Sort time-series data (Market Values).',
        file: 'interactive_notebook.py',
        content: <p>We strictly sort transfer history and market value logs by date to enable precise <code>merge_asof</code> later.</p>
    },
    {
        id: 'step_2_3',
        title: '2.3 Identify Top-5',
        desc: 'Find Premier League, LaLiga, etc.',
        file: 'interactive_notebook.py',
        content: <p>We scan the competitions table for standard codes (GB1, ES1, L1, IT1, FR1) to identify the Big 5 leagues.</p>
    },
    {
        id: 'step_2_4',
        title: '2.4 Apply Filter',
        desc: 'Drop players/games outside Big 5.',
        file: 'interactive_notebook.py',
        content: <p>We reduce the dataset to only rows associated with the identified Top-5 competitions.</p>
    },
    {
        id: 'step_2_5',
        title: '2.5 ID Gate',
        desc: 'Check overlap between Backbone and Injuries.',
        file: 'interactive_notebook.py',
        content: <p><strong>Critical Step:</strong> We assert that &gt;70% of our backbone players exist in the injury database. If not, the dataset is misaligned.</p>
    },

    // --- Phase 3 ---
    {
        id: 'step_3_1',
        title: '3.1 Temporal Grid',
        desc: 'Determine active weeks per season.',
        file: 'interactive_notebook.py',
        content: <p>We find the Min/Max date for each competition-season to define the valid simulation window.</p>
    },
    {
        id: 'step_3_3',
        title: '3.3 Build Skeleton',
        desc: 'Cross-join Active Players x Weeks.',
        file: 'interactive_notebook.py',
        content: <p>We create the empty <code>(player_id, week_start)</code> panel. This is the target index for our ML model.</p>
    },

    // --- Phase 4 ---
    {
        id: 'step_4_1',
        title: '4.1 Daily Aggregation',
        desc: 'Resample irregular matches to daily time-series.',
        file: 'interactive_notebook.py',
        content: (
            <div className="space-y-4">
                <p>Matches happen irregularly. To compute workloads, we first map them to a dense daily timeline.</p>
                <pre className="bg-gray-100 p-2 rounded text-xs font-mono">
                    {`# Resample to Daily, filling non-match days with 0 minutes
daily = series.resample('D').sum().fillna(0)`}
                </pre>
            </div>
        )
    },
    {
        id: 'step_4_2',
        title: '4.2 Rolling Sums',
        desc: 'Compute Workload (Last 7d, 28d).',
        file: 'interactive_notebook.py',
        content: <p>We calculate strictly prior rolling sums (7 days, 28 days) to quantify "Acute" and "Chronic" load.</p>
    },
    {
        id: 'step_4_3',
        title: '4.3 ACWR',
        desc: 'Calculate Acute:Chronic Workload Ratio.',
        file: 'interactive_notebook.py',
        content: <p><strong>ACWR</strong> = Load(7d) / (Load(28d) / 4). A value &gt; 1.5 suggests a spike in intensity.</p>
    },
    {
        id: 'step_4_4',
        title: '4.4 Congestion',
        desc: 'Match Density (matches in last 3/7 days).',
        file: 'interactive_notebook.py',
        content: <p>We count the number of matches played in recent windows to capture fixture congestion.</p>
    },

    // --- Phase 5 ---
    {
        id: 'step_5_1',
        title: '5.1 Generate Target',
        desc: 'Flag injury in next 30 days.',
        file: 'interactive_notebook.py',
        content: (
            <div className="space-y-4">
                <p>We check if an injury starts in the forward window <code>(t, t+30]</code>.</p>
                <pre className="bg-gray-100 p-2 rounded text-xs font-mono">
                    {`target = (injury_start > week_start) & 
         (injury_start <= week_start + 30d)`}
                </pre>
            </div>
        )
    },
    {
        id: 'step_5_3',
        title: '5.3 Finalize & Save',
        desc: 'Apply structural flags (Censoring) and save Parquet.',
        file: 'interactive_notebook.py',
        content: <p>We mask out the final 30 days of data (Right Censoring) and save the final <code>panel.parquet</code> for training.</p>
    },

    // --- Training ---
    {
        id: 'step_3_train',
        title: '6.0 Train Model',
        desc: 'Train HGBT on proposed panel.',
        file: 'train_model.py',
        content: <p>Finally, we run the machine learning training using the panel generated by the steps above.</p>
    }
];

function XtermConsole({ output }) {
    const terminalRef = useRef(null);
    const termInstance = useRef(null);

    useEffect(() => {
        if (!termInstance.current && terminalRef.current) {
            const term = new Terminal({
                rows: 24,
                cols: 80,
                theme: { background: '#1e1e1e', foreground: '#d4d4d4' },
                fontSize: 12,
                fontFamily: 'Menlo, Monaco, monospace'
            });
            const fitAddon = new FitAddon();
            term.loadAddon(fitAddon);
            term.open(terminalRef.current);
            fitAddon.fit();
            termInstance.current = term;
        }
    }, []);

    useEffect(() => {
        if (termInstance.current && output) {
            termInstance.current.clear();
            const formatted = output.replace(/\n/g, '\r\n');
            termInstance.current.write(formatted);
        }
    }, [output]);

    return <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid #334155' }} ref={terminalRef} />;
}

function Walkthrough() {
    const [activeStep, setActiveStep] = useState(0);
    const [loading, setLoading] = useState(false);
    const [logs, setLogs] = useState({});

    const runStep = async (stepIndex) => {
        const step = STEPS[stepIndex];
        setLoading(true);
        setLogs(prev => ({ ...prev, [step.id]: "Running..." }));

        try {
            const res = await axios.post('http://localhost:8000/api/run_step', { step_id: step.id });
            setLogs(prev => ({
                ...prev,
                [step.id]: (res.data.stdout || '') + (res.data.stderr ? '\n=== STDERR ===\n' + res.data.stderr : '')
            }));
        } catch (err) {
            setLogs(prev => ({ ...prev, [step.id]: `Error: ${err.message}` }));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
            <div className="header-container">
                <h1 className="page-title">Interactive Notebook Mode</h1>
                <p className="page-desc">Granular, step-by-step pipeline execution. Inspect every logical transformation.</p>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '2rem', flex: 1, minHeight: 0 }}>

                {/* Sidebar (Scrollable) */}
                <div style={{ overflowY: 'auto', paddingRight: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    {STEPS.map((step, idx) => {
                        const isActive = idx === activeStep;
                        const hasLog = !!logs[step.id];

                        return (
                            <div
                                key={step.id}
                                onClick={() => setActiveStep(idx)}
                                style={{
                                    padding: '1rem',
                                    borderRadius: '0.75rem',
                                    border: isActive ? '2px solid var(--primary)' : '1px solid var(--border)',
                                    backgroundColor: isActive ? 'var(--primary-light)' : 'white',
                                    cursor: 'pointer',
                                    transition: 'all 0.2s'
                                }}
                            >
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                                    <h3 style={{ fontWeight: 600, fontSize: '0.9rem', color: isActive ? 'var(--primary-hover)' : 'var(--text-primary)' }}>
                                        {step.title}
                                    </h3>
                                    {hasLog ? <CheckCircle size={16} className="text-green" /> : <Circle size={16} style={{ color: '#cbd5e1' }} />}
                                </div>
                                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.3, display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>{step.desc}</p>
                            </div>
                        );
                    })}
                </div>

                {/* Main Content Area (Exec + Doc) */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', minHeight: 0, overflowY: 'auto' }}>

                    {/* Execution Pane */}
                    <div className="card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <div>
                                <h2 className="section-title" style={{ marginBottom: '0.25rem' }}>
                                    {STEPS[activeStep].title}
                                </h2>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <span className="badge badge-gray" style={{ fontFamily: 'monospace' }}>
                                        {STEPS[activeStep].file}
                                    </span>
                                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>ID: {STEPS[activeStep].id}</span>
                                </div>
                            </div>
                            <button
                                onClick={() => runStep(activeStep)}
                                disabled={loading}
                                className="btn btn-primary"
                            >
                                {loading ? 'Executing...' : <><Play size={16} /> Run Step</>}
                            </button>
                        </div>

                        <div className="bg-dark rounded-lg p-4">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#94a3b8', fontSize: '0.75rem', marginBottom: '0.5rem', fontWeight: 600, letterSpacing: '0.05em' }}>
                                <TerminalIcon size={14} /> OUTPUT CONSOLE
                            </div>
                            <XtermConsole output={logs[STEPS[activeStep].id] || "Ready to execute..."} />
                        </div>
                    </div>

                    {/* Educational Content */}
                    <div className="card" style={{ flex: 1 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                            <BookOpen size={20} className="text-blue" />
                            <h3 className="section-title" style={{ marginBottom: 0 }}>Technical Details</h3>
                        </div>
                        <div style={{ lineHeight: '1.7', color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                            {STEPS[activeStep].content}
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
}

export default Walkthrough;
