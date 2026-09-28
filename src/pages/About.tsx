import React from 'react';
import { Camera, Cpu, Layers, ShieldCheck, Activity, ArrowRight } from 'lucide-react';

interface AboutProps {
  setCurrentPage: (page: string) => void;
}

export const About: React.FC<AboutProps> = ({ setCurrentPage }) => {
  return (
    <div className="space-y-12 py-8 max-w-4xl mx-auto text-slate-100">
      <div className="space-y-4 bg-slate-950 p-10 rounded-3xl border border-slate-800 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-cyan-500/15 to-transparent pointer-events-none"></div>
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-xs font-mono text-cyan-400">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          Project Story & Architecture
        </div>
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight">About Facial Mood Analysis</h1>
        <p className="text-sm text-slate-300 leading-relaxed max-w-2xl">
          Bridging single-camera computer vision with multi-representation feature fusion to establish reliable, real-time facial expression intelligence.
        </p>
      </div>

      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-6">
        <h3 className="text-xl font-bold text-white">The Research Motivation</h3>
        <p className="text-sm text-slate-300 leading-relaxed">
          Traditional facial emotion recognition systems rely heavily on raw RGB frames or single grayscale transformations. This approach is highly sensitive to illumination variance, pose shifts, and noise. Facial Mood Analysis introduces a robust multi-representation pipeline that derives multiple complementary views (Grayscale, CLAHE contrast, Canny edges, Sobel gradients, and LBP textures) from a single camera feed.
        </p>
        <p className="text-sm text-slate-300 leading-relaxed">
          By extracting 22 distinct statistical, textural, and probabilistic features and fusing them through a trained Logistic Regression classifier, the system achieves stable, real-time estimation across continuous video streams.
        </p>

        <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
          <span className="text-xs font-mono text-slate-400">Ready to test the workstation?</span>
          <button
            onClick={() => setCurrentPage('analyze')}
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-xl text-xs shadow-lg transition-all flex items-center gap-2"
          >
            Launch Analyze Workspace <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
