import React from 'react';

interface FooterProps {
  setCurrentPage: (page: string) => void;
}

export const Footer: React.FC<FooterProps> = ({ setCurrentPage }) => {
  return (
    <footer className="bg-slate-900 text-slate-400 border-t border-slate-800 py-12 px-6 mt-20">
      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <span className="font-bold text-white tracking-wider text-sm">FACIAL MOOD ANALYSIS</span>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Multi-representation real-time facial expression classification using single RGB input, FER2013 dataset, and 22D feature vector fusion.
          </p>
        </div>

        <div className="space-y-3 text-xs">
          <h4 className="font-mono uppercase tracking-wider text-slate-200">Navigation</h4>
          <ul className="space-y-2">
            <li><button onClick={() => setCurrentPage('home')} className="hover:text-cyan-400 transition-colors">Home</button></li>
            <li><button onClick={() => setCurrentPage('analyze')} className="hover:text-cyan-400 transition-colors">Analysis Workspace</button></li>
            <li><button onClick={() => setCurrentPage('how-it-works')} className="hover:text-cyan-400 transition-colors">How It Works</button></li>
            <li><button onClick={() => setCurrentPage('research')} className="hover:text-cyan-400 transition-colors">Research & Dataset</button></li>
          </ul>
        </div>

        <div className="space-y-3 text-xs">
          <h4 className="font-mono uppercase tracking-wider text-slate-200">Legal & Transparency</h4>
          <ul className="space-y-2">
            <li><button onClick={() => setCurrentPage('about')} className="hover:text-cyan-400 transition-colors">About Project</button></li>
            <li><button onClick={() => setCurrentPage('history')} className="hover:text-cyan-400 transition-colors">Analysis History</button></li>
            <li><button onClick={() => setCurrentPage('privacy')} className="hover:text-cyan-400 transition-colors">Privacy Policy</button></li>
            <li><button onClick={() => setCurrentPage('terms')} className="hover:text-cyan-400 transition-colors">Terms of Use</button></li>
          </ul>
        </div>

        <div className="space-y-3 text-xs">
          <h4 className="font-mono uppercase tracking-wider text-slate-200">System Specs</h4>
          <div className="p-3 bg-slate-800/80 rounded-xl border border-slate-700 font-mono text-[11px] space-y-1 text-slate-300">
            <div>Model: <span className="text-cyan-400">fer2013_linear</span></div>
            <div>Schema: <span className="text-cyan-400">v1 (22D)</span></div>
            <div>Smoothing: <span className="text-cyan-400">EMA (α=0.5)</span></div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto pt-8 border-t border-slate-800/80 flex flex-col md:flex-row items-center justify-between text-xs text-slate-500 gap-4">
        <div>© 2026 Facial Mood Analysis Research Group. All rights reserved.</div>
        <div>Strictly for research and educational purposes. Not for medical diagnosis.</div>
      </div>
    </footer>
  );
};
