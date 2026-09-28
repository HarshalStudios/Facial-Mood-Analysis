import React from 'react';
import { Camera, Eye, Cpu, Layers, BarChart3, CheckCircle2 } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const steps = [
    { num: '01', title: 'Single RGB Camera Input', desc: 'Acquires a standard BGR uint8 frame from the connected webcam or uploaded video file.', icon: Camera },
    { num: '02', title: 'Face Detection & Cropping', desc: 'Detects facial bounding boxes using OpenCV detectors and crops the primary face region.', icon: Eye },
    { num: '03', title: 'Multiple Visual Representations', desc: 'Generates 6 distinct representation views: Grayscale, CLAHE contrast enhancement, Canny edges, Sobel gradient energy, and LBP texture histograms.', icon: Layers },
    { num: '04', title: '22-Dimensional Feature Extraction', desc: 'Extracts 7 emotion probability scores, 10 LBP histogram bins, edge density, gradient energy, brightness, contrast, and Laplacian sharpness.', icon: BarChart3 },
    { num: '05', title: 'Feature-Level Fusion Model', desc: 'Applies StandardScaler normalization and a supervised LogisticRegression classifier trained on FER2013.', icon: Cpu },
    { num: '06', title: 'Temporal Smoothing & Quality Check', desc: 'Applies EMA temporal smoothing across a 5-frame buffer to eliminate prediction jitter.', icon: CheckCircle2 }
  ];

  return (
    <div className="space-y-12 py-8 max-w-4xl mx-auto text-slate-100">
      <div className="text-center space-y-3">
        <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">Architecture Pipeline</span>
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-white">How Facial Mood Analysis Works</h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto">
          A step-by-step walk-through of the multi-representation computer vision and machine learning pipeline.
        </p>
      </div>

      <div className="space-y-6 relative before:absolute before:inset-0 before:left-8 before:w-0.5 before:bg-slate-800">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <div key={idx} className="relative flex items-start gap-6 bg-slate-900/80 backdrop-blur-xl p-8 rounded-3xl border border-slate-800 shadow-xl ml-4 md:ml-0 hover:border-cyan-500/50 transition-colors">
              <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center shrink-0 font-mono font-bold text-sm">
                {step.num}
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <h3 className="text-lg font-bold text-white">{step.title}</h3>
                </div>
                <p className="text-sm text-slate-400 leading-relaxed">{step.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
