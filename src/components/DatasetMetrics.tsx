import React, { useEffect, useState } from 'react';
import { Database, Award, CheckCircle, TrendingUp } from 'lucide-react';

interface Experiment {
  experiment: string;
  test_accuracy: number;
  test_macro_f1: number;
}

export const DatasetMetrics: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    fetch('/api/experiments')
      .then(res => res.json())
      .then(data => setExperiments(data))
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Dataset Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">Dataset Source</span>
          <span className="text-xl font-bold text-slate-900 block">FER2013</span>
          <span className="text-xs text-slate-500 mt-1 block">48x48 Grayscale Images</span>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">Training Split</span>
          <span className="text-xl font-mono font-bold text-slate-900 block">28,709</span>
          <span className="text-xs text-slate-500 mt-1 block">Stratified Training Samples</span>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">Validation Split</span>
          <span className="text-xl font-mono font-bold text-slate-900 block">3,589</span>
          <span className="text-xs text-slate-500 mt-1 block">Public Test Split</span>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
          <span className="text-xs font-mono uppercase tracking-wider text-slate-400 block mb-1">Test Accuracy</span>
          <span className="text-xl font-mono font-bold text-cyan-600 block">37.61%</span>
          <span className="text-xs text-slate-500 mt-1 block">Macro F1: 30.76%</span>
        </div>
      </div>

      {/* Phase 9 Ablation Experiments Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-slate-900">Phase 9 Ablation Experiments</h3>
            <p className="text-xs text-slate-500">Evaluated on real FER2013-derived feature data using Logistic Regression fusion</p>
          </div>
          <span className="text-xs font-mono text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200">7 Experiments Registered</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-mono text-slate-500 bg-slate-50/50">
                <th className="px-6 py-3 font-semibold">Experiment Configuration</th>
                <th className="px-6 py-3 font-semibold text-right">Test Accuracy</th>
                <th className="px-6 py-3 font-semibold text-right">Test Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-sm">
              {experiments.map((exp, idx) => (
                <tr key={idx} className="hover:bg-slate-50/50 transition-colors">
                  <td className="px-6 py-3.5 font-medium text-slate-900 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-600"></span>
                    {exp.experiment}
                  </td>
                  <td className="px-6 py-3.5 font-mono text-right text-slate-700">
                    {(exp.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="px-6 py-3.5 font-mono text-right font-bold text-cyan-700">
                    {(exp.test_macro_f1 * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
