import React, { useEffect, useState } from 'react';
import { Database, Award, Cpu, Layers, ShieldAlert, CheckCircle2 } from 'lucide-react';

interface Experiment {
  experiment: string;
  test_accuracy: number;
  test_macro_f1: number;
}

const DEFAULT_EXPERIMENTS: Experiment[] = [
  { experiment: "Emotion-probability baseline", test_accuracy: 0.3639, test_macro_f1: 0.2972 },
  { experiment: "+ LBP", test_accuracy: 0.3711, test_macro_f1: 0.3022 },
  { experiment: "+ Edge", test_accuracy: 0.3700, test_macro_f1: 0.3027 },
  { experiment: "+ Gradient", test_accuracy: 0.3697, test_macro_f1: 0.3053 },
  { experiment: "+ Image Quality", test_accuracy: 0.3703, test_macro_f1: 0.3026 },
  { experiment: "+ All Handcrafted", test_accuracy: 0.3761, test_macro_f1: 0.3076 },
  { experiment: "Handcrafted Only", test_accuracy: 0.2803, test_macro_f1: 0.1746 }
];

export const Research: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>(DEFAULT_EXPERIMENTS);

  useEffect(() => {
    fetch('/api/experiments')
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch experiments');
        return res.json();
      })
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          setExperiments(data);
        }
      })
      .catch(() => {
        // Fallback to DEFAULT_EXPERIMENTS gracefully
      });
  }, []);

  return (
    <div className="space-y-12 py-8 max-w-5xl mx-auto text-slate-100">
      {/* Header */}
      <div className="space-y-4 bg-slate-950 p-10 rounded-3xl border border-slate-800 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-cyan-500/15 to-transparent pointer-events-none"></div>
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-xs font-mono text-cyan-400">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          Research & Dataset Transparency
        </div>
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight">Dataset, Models & Architecture</h1>
        <p className="text-sm text-slate-300 leading-relaxed max-w-3xl">
          The Facial Mood Analysis system addresses the limitations of single-representation emotion classifiers by transforming a single RGB input into multiple calibrated visual representations and fusing their features into a 22-dimensional feature vector.
        </p>
      </div>

      {/* Dataset & Model Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-3xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center font-bold">
            <Database className="w-6 h-6" />
          </div>
          <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Primary Dataset</span>
          <h3 className="text-xl font-bold text-white">FER2013</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            35,887 grayscale facial expression images (48×48 pixels) across 7 canonical expression classes.
          </p>
          <div className="pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400 space-y-1.5">
            <div className="flex justify-between"><span>Training Split:</span><span className="font-semibold text-slate-200">28,709 images</span></div>
            <div className="flex justify-between"><span>Validation Split:</span><span className="font-semibold text-slate-200">3,589 images</span></div>
            <div className="flex justify-between"><span>Test Split:</span><span className="font-semibold text-slate-200">3,589 images</span></div>
          </div>
        </div>

        <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-3xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center font-bold">
            <Cpu className="w-6 h-6" />
          </div>
          <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Model Architecture</span>
          <h3 className="text-xl font-bold text-white">Logistic Regression</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Supervised fusion classifier trained on StandardScaler-normalized 22-dimensional feature vectors.
          </p>
          <div className="pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400 space-y-1.5">
            <div className="flex justify-between"><span>Feature Schema:</span><span className="font-semibold text-slate-200">v1 (22D)</span></div>
            <div className="flex justify-between"><span>Provider:</span><span className="font-semibold text-slate-200">fer2013_linear</span></div>
            <div className="flex justify-between"><span>Temporal Smoothing:</span><span className="font-semibold text-slate-200">EMA (α=0.5)</span></div>
          </div>
        </div>

        <div className="bg-slate-900/80 backdrop-blur-xl p-6 rounded-3xl border border-slate-800 shadow-xl space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center font-bold">
            <Award className="w-6 h-6" />
          </div>
          <span className="text-xs font-mono uppercase text-slate-400 tracking-wider">Test Performance</span>
          <h3 className="text-xl font-bold text-cyan-400">37.61% Accuracy</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Macro F1 score of 30.76% achieved using all handcrafted and probability features combined.
          </p>
          <div className="pt-3 border-t border-slate-800 text-[11px] font-mono text-slate-400 space-y-1.5">
            <div className="flex justify-between"><span>Classes:</span><span className="font-semibold text-slate-200">7 Canonical</span></div>
            <div className="flex justify-between"><span>Baseline Accuracy:</span><span className="font-semibold text-slate-200">36.39%</span></div>
          </div>
        </div>
      </div>

      {/* 7 Canonical Classes */}
      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-4">
        <h3 className="text-base font-bold text-white">7 Canonical Expression Classes</h3>
        <p className="text-xs text-slate-400">The model evaluates probability distribution across standard emotion categories:</p>
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-3">
          {['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'].map((emo, idx) => (
            <div key={idx} className="p-4 bg-slate-950 rounded-2xl border border-slate-800 text-center">
              <span className="text-xs font-mono font-bold text-cyan-400 uppercase block">{emo}</span>
              <span className="text-[10px] text-slate-500 mt-1 block">Class {idx+1}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Phase 9 Ablation Table */}
      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 overflow-hidden shadow-xl">
        <div className="px-8 py-5 border-b border-slate-800 bg-slate-950/50">
          <h3 className="text-base font-bold text-white">Phase 9 Ablation Experiment Results</h3>
          <p className="text-xs text-slate-400">Evaluating feature contribution across real FER2013-derived test splits</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 text-xs font-mono text-slate-400 bg-slate-950/30">
                <th className="px-8 py-3.5 font-semibold">Experiment Configuration</th>
                <th className="px-8 py-3.5 font-semibold text-right">Test Accuracy</th>
                <th className="px-8 py-3.5 font-semibold text-right">Test Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-sm">
              {experiments.map((exp, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-8 py-4 font-medium text-slate-200">{exp.experiment}</td>
                  <td className="px-8 py-4 font-mono text-right text-slate-300">{(exp.test_accuracy * 100).toFixed(2)}%</td>
                  <td className="px-8 py-4 font-mono text-right font-bold text-cyan-400">{(exp.test_macro_f1 * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Limitations & Transparency */}
      <div className="bg-amber-500/10 rounded-3xl border border-amber-500/20 p-8 space-y-3">
        <div className="flex items-center gap-2 text-amber-400 font-bold text-sm">
          <ShieldAlert className="w-4 h-4" /> Model Limitations & Ethical Notice
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          FER2013 images are 48×48 grayscale representations captured under varying real-world conditions, leading to inherent classification challenges. Predictions represent statistical estimates and probabilities derived from computer vision features. This software is designed strictly for research, educational, and demonstration purposes and must not be used for medical, psychological, or psychiatric diagnosis.
        </p>
      </div>
    </div>
  );
};
