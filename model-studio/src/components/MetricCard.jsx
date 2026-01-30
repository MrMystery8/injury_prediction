import React from 'react';

export function MetricCard({ title, value, subtext, trend }) {
    const isPositive = trend > 0;

    const cardStyle = {
        backgroundColor: 'white',
        padding: '1.5rem',
        borderRadius: '0.5rem',
        boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        border: '1px solid #e5e7eb',
        display: 'flex',
        flexDirection: 'column'
    };

    const titleStyle = {
        color: '#6b7280',
        fontSize: '0.875rem',
        fontWeight: 500,
        textTransform: 'uppercase',
        letterSpacing: '0.05em',
        marginBottom: '0.5rem'
    };

    const valueContainerStyle = {
        display: 'flex',
        alignItems: 'baseline'
    };

    const valueStyle = {
        fontSize: '1.875rem',
        fontWeight: 700,
        color: '#111827'
    };

    const trendStyle = {
        marginLeft: '0.5rem',
        fontSize: '0.875rem',
        fontWeight: 500,
        color: isPositive ? '#059669' : '#dc2626'
    };

    const subtextStyle = {
        marginTop: '0.25rem',
        fontSize: '0.875rem',
        color: '#9ca3af'
    };

    return (
        <div style={cardStyle}>
            <h3 style={titleStyle}>{title}</h3>
            <div style={valueContainerStyle}>
                <span style={valueStyle}>{value}</span>
                {trend && (
                    <span style={trendStyle}>
                        {isPositive ? '+' : ''}{trend}%
                    </span>
                )}
            </div>
            {subtext && <p style={subtextStyle}>{subtext}</p>}
        </div>
    );
}
