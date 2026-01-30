import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, PlayCircle, Settings, Activity } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import DataExplorer from './pages/DataExplorer';
import Walkthrough from './pages/Walkthrough';
import TrainingLab from './pages/TrainingLab';
import './App.css';

function NavLink({ to, icon: Icon, label }) {
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <Link to={to} className={`nav-link ${isActive ? 'active' : ''}`}>
      <Icon size={20} />
      <span>{label}</span>
      {isActive && <div className="active-indicator" />}
    </Link>
  );
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand">
        <Activity className="icon-blue" size={28} />
        <span className="brand-text">Model Studio</span>
      </div>

      <nav className="nav-menu">
        <NavLink to="/" icon={LayoutDashboard} label="Dashboard" />
        <NavLink to="/data" icon={Database} label="Data Explorer" />
        <NavLink to="/walkthrough" icon={PlayCircle} label="Walkthrough" />
        <NavLink to="/lab" icon={Settings} label="Training Lab" />
      </nav>

      <div className="sidebar-footer">
        <div className="status-dot online"></div>
        <span>Server Connected</span>
      </div>
    </aside>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app-layout">
        <Sidebar />
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/walkthrough" element={<Walkthrough />} />
            <Route path="/lab" element={<TrainingLab />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
