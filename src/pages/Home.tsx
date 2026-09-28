import React from 'react';
import { 
  Camera, ArrowRight, Layers, Cpu, BarChart3, Activity, 
  ShieldCheck, Zap, Sparkles, Eye, Compass, Film, Award, CheckCircle2
} from 'lucide-react';
import { PipelineBar } from '../components/PipelineBar';

interface HomeProps {
  setCurrentPage: (page: string) => void;
}

export const Home: React.FC<HomeProps> = ({ setCurrentPage }) => {
  return (
    <div className="space-y-24 py-4 w-full">
      {/* Editorial Hero (Inspired by Pixel Rise composition with retro-futuristic CV Lab aesthetic) */}
      <section className="relative overflow-hidden bg-gradient-to-b from-slate-900 via-slate-950 to-slate-900 text-white border border-slate-800/80 rounded-[2.5rem] p-6 md:p-16 shadow-2xl">
        {/* Background glow & subtle tech grid */}
        <div className="absolute inset-0 bg-[radial-gradient(#06b6d4_1px,transparent_1px)] [background-size:32px_32px] opacity-15 pointer-events-none"></div>
        <div className="absolute -top-32 -right-32 w-[600px] h-[600px] bg-cyan-500/15 rounded-full blur-[140px] pointer-events-none"></div>
        <div className="absolute -bottom-32 -left-32 w-[500px] h-[500px] bg-sky-500/10 rounded-full blur-[140px] pointer-events-none"></div>

        {/* Big Stylized Background Typography */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full text-center pointer-events-none select-none opacity-[0.04] font-black text-[14vw] tracking-tighter text-white">
          VISION LAB
        </div>

        {/* Top Header Row in Hero */}
        <div className="relative z-10 flex flex-wrap items-center justify-between gap-4 mb-10 pb-6 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="px-3.5 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-xs font-mono font-bold text-cyan-400 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              [1/6] MULTI-REPRESENTATION CV ENGINE
            </div>
            <span className="hidden sm:inline text-xs font-mono text-slate-400">FER2013 · 22D FEATURE FUSION</span>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-slate-300">
            <span className="flex items-center gap-1.5 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> 100% Client-Local Stream
            </span>
          </div>
        </div>

        {/* Main Hero Grid */}
        <div className="relative z-10 grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          {/* Left Editorial Copy */}
          <div className="lg:col-span-6 space-y-6">
            <h1 className="text-4xl sm:text-6xl md:text-7xl font-black tracking-tight leading-[1.02] text-white">
              BUILDING <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-400">
                MULTIVIEW CV
              </span> <br />
              THAT SEES DEEPER.
            </h1>

            <p className="text-base sm:text-lg text-slate-300 leading-relaxed font-normal max-w-xl">
              One standard RGB camera feed simultaneously transformed into six calibrated representations—fused through a 22-dimensional feature vector for robust, real-time facial mood estimation.
            </p>

            <div className="flex flex-wrap items-center gap-4 pt-2">
              <button
                onClick={() => setCurrentPage('analyze')}
                className="px-8 py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-black rounded-2xl shadow-xl shadow-cyan-500/25 transition-all flex items-center gap-3 group text-sm tracking-wide"
              >
                Launch Workspace <ArrowRight className="w-4 h-4 group-hover:translate-x-1.5 transition-transform" />
              </button>
              <button
                onClick={() => setCurrentPage('how-it-works')}
                className="px-7 py-4 bg-slate-800/80 hover:bg-slate-800 text-white font-bold rounded-2xl transition-colors text-sm border border-slate-700/80 backdrop-blur-xl"
              >
                Architecture Guide
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 pt-6 border-t border-slate-800/60 text-xs font-mono">
              <div className="p-3 bg-slate-900/60 rounded-2xl border border-slate-800">
                <span className="text-cyan-400 font-extrabold text-base sm:text-xl block">35.8k</span>
                <span className="text-slate-400 text-[10px]">FER2013 Dataset</span>
              </div>
              <div className="p-3 bg-slate-900/60 rounded-2xl border border-slate-800">
                <span className="text-white font-extrabold text-base sm:text-xl block">22D</span>
                <span className="text-slate-400 text-[10px]">Fused Vector</span>
              </div>
              <div className="p-3 bg-slate-900/60 rounded-2xl border border-slate-800">
                <span className="text-emerald-400 font-extrabold text-base sm:text-xl block">7 Classes</span>
                <span className="text-slate-400 text-[10px]">Canonical Moods</span>
              </div>
            </div>
          </div>

          {/* Right Floating Visual Centerpiece (Inspired by Retro 3D Terminal & Camera Rig) */}
          <div className="lg:col-span-6 relative flex items-center justify-center">
            {/* Ambient circular glow */}
            <div className="w-[320px] sm:w-[420px] aspect-square rounded-full bg-gradient-to-tr from-cyan-500/20 via-sky-500/10 to-amber-500/10 border border-cyan-500/20 absolute blur-xl"></div>

            {/* Central Terminal Mockup Card */}
            <div className="relative w-full max-w-md bg-slate-950/90 border border-slate-700/80 rounded-3xl p-5 shadow-2xl backdrop-blur-2xl space-y-4">
              {/* Terminal Titlebar */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-rose-500/80"></span>
                  <span className="w-3 h-3 rounded-full bg-amber-500/80"></span>
                  <span className="w-3 h-3 rounded-full bg-emerald-500/80"></span>
                  <span className="text-[11px] font-mono text-slate-400 ml-2">rgb_fusion_v1.sh</span>
                </div>
                <div className="px-2 py-0.5 bg-cyan-500/10 text-cyan-400 text-[10px] font-mono rounded border border-cyan-500/30 font-semibold">
                  LIVE CV RIG
                </div>
              </div>

              {/* 6-Representation Matrix Mini Preview */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { name: 'RGB Live', col: 'from-cyan-500/30 to-blue-500/30', border: 'border-cyan-500/40' },
                  { name: 'Grayscale', col: 'from-slate-700/40 to-slate-800/40', border: 'border-slate-700' },
                  { name: 'CLAHE', col: 'from-sky-700/30 to-slate-800/40', border: 'border-sky-500/30' },
                  { name: 'Canny', col: 'from-cyan-900/40 to-slate-900/60', border: 'border-cyan-500/30' },
                  { name: 'Sobel', col: 'from-indigo-900/40 to-slate-900/60', border: 'border-indigo-500/30' },
                  { name: 'LBP Hist', col: 'from-teal-900/40 to-slate-900/60', border: 'border-teal-500/30' }
                ].map((m, i) => (
                  <div key={i} className={`h-16 rounded-xl bg-gradient-to-br ${m.col} border ${m.border} p-2 flex flex-col justify-between relative overflow-hidden group`}>
                    <span className="text-[9px] font-mono font-bold text-slate-300">{m.name}</span>
                    <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse self-end"></div>
                  </div>
                ))}
              </div>

              {/* Real-time Telemetry Bar */}
              <div className="p-3 bg-slate-900/80 rounded-2xl border border-slate-800 flex items-center justify-between text-[11px] font-mono">
                <span className="text-slate-400">Fused Output:</span>
                <span className="text-cyan-400 font-extrabold flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" /> HAPPY (84.2%)
                </span>
              </div>
            </div>

            {/* Floating Metric Pill 1 (Top Right) */}
            <div className="absolute -top-4 -right-2 sm:right-2 p-3 bg-slate-900/90 border border-slate-700 rounded-2xl shadow-xl backdrop-blur-xl flex items-center gap-3 animate-bounce [animation-duration:4s]">
              <div className="w-9 h-9 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold text-xs">
                22D
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-400 block">Vector Schema</span>
                <span className="text-xs font-bold text-white">Full Feature Fusion</span>
              </div>
            </div>

            {/* Floating Metric Pill 2 (Bottom Left) */}
            <div className="absolute -bottom-6 -left-2 sm:left-2 p-3 bg-slate-900/90 border border-slate-700 rounded-2xl shadow-xl backdrop-blur-xl flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs">
                EMA
              </div>
              <div>
                <span className="text-[10px] font-mono text-slate-400 block">Temporal Jitter Filter</span>
                <span className="text-xs font-bold text-white">α = 0.50 Smoothing</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Signature Pipeline Component */}
      <section className="space-y-6">
        <div className="text-center space-y-2 max-w-xl mx-auto">
          <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">End-to-End Architecture</span>
          <h2 className="text-3xl font-extrabold tracking-tight text-white">Visual Pipeline Flow</h2>
        </div>
        <PipelineBar />
      </section>

      {/* Modern Feature Cards */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-4 hover:border-cyan-500/50 transition-colors">
          <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold">
            <Layers className="w-7 h-7" />
          </div>
          <h3 className="text-xl font-bold text-white">Multi-Representation DIP</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Extracts Grayscale, CLAHE contrast enhancement, Canny edges, Sobel gradient energy, and LBP texture maps simultaneously from every single input frame.
          </p>
        </div>

        <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-4 hover:border-cyan-500/50 transition-colors">
          <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold">
            <Cpu className="w-7 h-7" />
          </div>
          <h3 className="text-xl font-bold text-white">Supervised Feature Fusion</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Combines 7 supervised emotion probabilities with 15 handcrafted texture and edge metrics into a unified 22-dimensional feature vector.
          </p>
        </div>

        <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-4 hover:border-cyan-500/50 transition-colors">
          <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center font-bold">
            <BarChart3 className="w-7 h-7" />
          </div>
          <h3 className="text-xl font-bold text-white">Temporal Stability</h3>
          <p className="text-sm text-slate-400 leading-relaxed">
            Stateful Exponential Moving Average (EMA) smoothing eliminates prediction jitter across continuous frames while validating quality metrics.
          </p>
        </div>
      </section>
    </div>
  );
};
