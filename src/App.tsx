import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { Analyze } from './pages/Analyze';
import { HowItWorks } from './pages/HowItWorks';
import { Research } from './pages/Research';
import { About } from './pages/About';
import { History } from './pages/History';
import { Privacy } from './pages/Privacy';
import { Terms } from './pages/Terms';

export function App() {
  const [currentPage, setCurrentPage] = useState('home');
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('mood_analysis_theme');
    if (saved === 'dark' || saved === 'light') return saved;
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  });

  useEffect(() => {
    localStorage.setItem('mood_analysis_theme', theme);
    if (theme === 'light') {
      document.documentElement.classList.add('light');
    } else {
      document.documentElement.classList.remove('light');
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  return (
    <div className={`min-h-screen flex flex-col font-sans transition-colors duration-300 ${
      theme === 'light' ? 'bg-slate-100 text-slate-900' : 'bg-slate-950 text-slate-100'
    }`}>
      <Header 
        currentPage={currentPage} 
        setCurrentPage={setCurrentPage} 
        theme={theme} 
        toggleTheme={toggleTheme} 
      />
      
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        {currentPage === 'home' && <Home setCurrentPage={setCurrentPage} />}
        {currentPage === 'analyze' && <Analyze />}
        {currentPage === 'how-it-works' && <HowItWorks />}
        {currentPage === 'research' && <Research />}
        {currentPage === 'about' && <About setCurrentPage={setCurrentPage} />}
        {currentPage === 'history' && <History />}
        {currentPage === 'privacy' && <Privacy />}
        {currentPage === 'terms' && <Terms />}
      </main>

      <Footer setCurrentPage={setCurrentPage} />
    </div>
  );
}

export default App;
