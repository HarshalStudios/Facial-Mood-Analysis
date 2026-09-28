import React, { useState } from 'react';
import { Code, Send, CheckCircle2, Play } from 'lucide-react';

export const ApiInspector: React.FC = () => {
  const [endpoint, setEndpoint] = useState('/api/status');
  const [method, setMethod] = useState<'GET' | 'POST'>('GET');
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const testEndpoint = async (ep: string, m: 'GET' | 'POST') => {
    setEndpoint(ep);
    setMethod(m);
    setLoading(true);
    try {
      const res = await fetch(ep, {
        method: m,
        headers: m === 'POST' ? { 'Content-Type': 'application/json' } : undefined,
        body: m === 'POST' ? JSON.stringify({ image: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==' }) : undefined
      });
      const data = await res.json();
      setResponse(data);
    } catch (err: any) {
      setResponse({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left: Endpoint List */}
      <div className="lg:col-span-5 bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-4">
        <div>
          <h2 className="text-base font-bold text-slate-900">API Contract Endpoints</h2>
          <p className="text-xs text-slate-500">Test live backend endpoints conforming to the system specification</p>
        </div>

        <div className="space-y-2">
          {[
            { path: '/api/health', method: 'GET', desc: 'System health check' },
            { path: '/api/ready', method: 'GET', desc: 'Model readiness status' },
            { path: '/api/status', method: 'GET', desc: 'System configuration & metadata' },
            { path: '/api/experiments', method: 'GET', desc: 'Phase 9 ablation results' },
            { path: '/api/predict/image', method: 'POST', desc: 'Raw image inference' },
            { path: '/api/predict/temporal/image', method: 'POST', desc: 'Temporal smoothing & quality pipeline' }
          ].map((item, idx) => (
            <div
              key={idx}
              onClick={() => testEndpoint(item.path, item.method as 'GET' | 'POST')}
              className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                endpoint === item.path ? 'bg-cyan-50 border-cyan-300' : 'bg-slate-50 border-slate-200 hover:border-slate-300'
              }`}
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                    item.method === 'GET' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                  }`}>
                    {item.method}
                  </span>
                  <span className="font-mono text-xs font-bold text-slate-800">{item.path}</span>
                </div>
                <p className="text-xs text-slate-500 mt-1">{item.desc}</p>
              </div>
              <Play className="w-4 h-4 text-cyan-600" />
            </div>
          ))}
        </div>
      </div>

      {/* Right: Response Output */}
      <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 p-6 shadow-sm flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className={`px-2 py-0.5 rounded text-xs font-mono font-bold ${
              method === 'GET' ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
            }`}>
              {method}
            </span>
            <span className="font-mono text-xs font-bold text-slate-800">{endpoint}</span>
          </div>
          <button
            onClick={() => testEndpoint(endpoint, method)}
            disabled={loading}
            className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-700 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 disabled:opacity-50"
          >
            <Send className="w-3.5 h-3.5" /> {loading ? 'Executing...' : 'Send Request'}
          </button>
        </div>

        <div className="flex-1 bg-slate-900 rounded-lg p-4 font-mono text-xs text-cyan-400 overflow-auto max-h-[400px]">
          <pre>{response ? JSON.stringify(response, null, 2) : '// Select an endpoint on the left and click Send Request'}</pre>
        </div>
      </div>
    </div>
  );
};
