import React, { useState, useEffect } from 'react';
import { Camera, Sun, Moon, Menu, X, Activity, ArrowRight } from 'lucide-react';

interface HeaderProps {
  currentPage: string;
  setCurrentPage: (page: string) => void;
  theme: 'dark' | 'light';
  toggleTheme: () => void;
}

export const Header: React.FC<HeaderProps> = ({ currentPage, setCurrentPage, theme, toggleTheme }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'analyze', label: 'Analyze' },
    { id: 'how-it-works', label: 'How It Works' },
    { id: 'research', label: 'Research' },
    { id: 'history', label: 'History' },
  ];

  return (
    <header className={`sticky top-0 z-50 transition-all duration-300 ${
      scrolled 
        ? 'bg-slate-900/80 dark:bg-slate-950/80 backdrop-blur-xl border-b border-slate-800 shadow-2xl py-3' 
        : 'bg-transparent py-5'
    }`}>
      <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
        {/* Brand Logo */}
        <button 
          onClick={() => setCurrentPage('home')}
          className="flex items-center gap-3 group text-left"
        >
          <div className="w-9 h-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-105 transition-transform">
            <Camera className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Vision Lab</span>
            </div>
            <span className="text-sm md:text-base font-extrabold tracking-tight text-white">FACIAL MOOD ANALYSIS</span>
          </div>
        </button>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1 bg-slate-800/50 dark:bg-slate-900/50 p-1 rounded-2xl border border-slate-700/60 backdrop-blur-md">
          {navItems.map((item) => {
            const isActive = currentPage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setCurrentPage(item.id)}
                className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all relative ${
                  isActive 
                    ? 'bg-cyan-500 text-slate-950 shadow-md font-bold' 
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/80'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Right Actions */}
        <div className="hidden md:flex items-center gap-3">
          <button
            onClick={toggleTheme}
            className="w-10 h-10 rounded-xl bg-slate-800/80 border border-slate-700 text-slate-300 hover:text-white flex items-center justify-center transition-colors"
            title="Toggle Theme"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-cyan-400" />}
          </button>

          <button
            onClick={() => setCurrentPage('analyze')}
            className="px-5 py-2.5 bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold rounded-xl shadow-lg shadow-cyan-500/20 transition-all flex items-center gap-2 group"
          >
            Start Analysis <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>

        {/* Mobile Menu Button */}
        <div className="flex md:hidden items-center gap-2">
          <button
            onClick={toggleTheme}
            className="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center"
          >
            {theme === 'dark' ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-cyan-400" />}
          </button>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="w-10 h-10 rounded-xl bg-slate-800 border border-slate-700 text-white flex items-center justify-center"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden absolute top-full left-0 w-full bg-slate-950/95 border-b border-slate-800 p-6 backdrop-blur-2xl space-y-4 animate-in slide-in-from-top duration-200">
          <div className="flex flex-col space-y-2">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => { setCurrentPage(item.id); setMobileMenuOpen(false); }}
                className={`px-4 py-3 rounded-xl text-left text-sm font-semibold transition-all ${
                  currentPage === item.id ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-300 hover:bg-slate-900'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <button
            onClick={() => { setCurrentPage('analyze'); setMobileMenuOpen(false); }}
            className="w-full py-3 bg-cyan-500 text-slate-950 font-bold rounded-xl text-sm text-center shadow-lg"
          >
            Start Analysis Workspace
          </button>
        </div>
      )}
    </header>
  );
};
