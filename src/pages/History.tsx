import React, { useState, useEffect } from 'react';
import { History as HistoryIcon, Trash2, Clock, Activity, ArrowRight, RefreshCw, CheckCircle2 } from 'lucide-react';

interface HistoryItem {
  timestamp: string;
  emotion: string;
  confidence: number;
  frame_id: number;
  frame_hash?: string;
}

export const History: React.FC = () => {
  const [items, setItems] = useState<HistoryItem[]>([]);

  const loadHistory = () => {
    const saved = localStorage.getItem('mood_analysis_history');
    console.log("[HISTORY] loading:", saved);
    if (saved) {
      try {
        setItems(JSON.parse(saved));
      } catch (e) {
        setItems([]);
      }
    } else {
      setItems([]);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const clearHistory = () => {
    localStorage.removeItem('mood_analysis_history');
    setItems([]);
    console.log("[HISTORY] cleared mood_analysis_history");
  };

  return (
    <div className="space-y-8 py-8 max-w-4xl mx-auto text-slate-100">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-slate-950 p-8 rounded-3xl border border-slate-800 shadow-2xl">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Analysis Archive</span>
            <span className="text-[10px] font-mono px-2 py-0.5 bg-slate-800 text-slate-300 rounded border border-slate-700">
              Engine v2.1.0-live
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Prediction History</h1>
          <p className="text-xs text-slate-400">Locally stored history of live facial expression predictions.</p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadHistory}
            className="px-3.5 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-xl text-xs font-semibold transition-colors flex items-center gap-1.5"
            title="Refresh History from localStorage"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
          <button
            onClick={clearHistory}
            className="px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/40 rounded-xl text-xs font-bold transition-colors flex items-center gap-2 shadow-sm"
          >
            <Trash2 className="w-4 h-4" /> CLEAR TEST HISTORY
          </button>
        </div>
      </div>

      {items.length === 0 ? (
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-16 text-center space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-slate-800 text-slate-500 flex items-center justify-center mx-auto border border-slate-700">
            <Clock className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-white">No Analysis Records Found</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Prediction history is clean. Run live webcam predictions in the Analyze Workspace to populate fresh records.
          </p>
        </div>
      ) : (
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl">
          <div className="p-4 bg-slate-950/60 border-b border-slate-800 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>Total Records: <strong className="text-white">{items.length}</strong></span>
            <span>Storage Key: <code className="text-cyan-400">mood_analysis_history</code></span>
          </div>
          <div className="divide-y divide-slate-800">
            {items.map((item, idx) => (
              <div key={idx} className="p-5 flex items-center justify-between hover:bg-slate-800/40 transition-colors">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold text-xs font-mono">
                    #{item.frame_id || idx+1}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-bold text-white capitalize">{item.emotion}</h4>
                      {item.frame_hash && (
                        <span className="text-[10px] font-mono px-1.5 py-0.2 bg-slate-800 text-slate-400 rounded">
                          hash:{item.frame_hash}
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] font-mono text-slate-400">{new Date(item.timestamp).toLocaleString()}</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-cyan-400">{Math.round(item.confidence * 100)}%</span>
                  <span className="text-[10px] text-slate-500 block">Confidence</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
