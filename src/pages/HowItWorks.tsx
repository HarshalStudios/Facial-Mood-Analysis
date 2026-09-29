import React from 'react';
import { Camera, Eye, Cpu, Layers, BarChart3, CheckCircle2, Scan, ShieldCheck, Sparkles, AlertCircle } from 'lucide-react';

export const HowItWorks: React.FC = () => {
  const steps = [
    { 
      num: '01', 
      title: 'Single RGB Camera Input', 
      desc: 'Acquires a standard BGR uint8 frame from the connected webcam or uploaded video file at high frame rates.', 
      icon: Camera 
    },
    { 
      num: '02', 
      title: 'Face Detection & Spatial Localization (Not Identification)', 
      desc: 'Detects facial bounding boxes and landmark anchors strictly to localize the face region. IMPORTANT: The system performs face localization for expression analysis only; it does NOT perform biometric identity recognition or person identification.', 
      icon: Eye 
    },
    { 
      num: '03', 
      title: 'Multiple Visual DIP Representations', 
      desc: 'Generates 6 distinct representation views: Grayscale (ITU-R BT.601), CLAHE contrast equalization, Canny spatial edges, Sobel gradient energy field, and LBP texture micro-patterns.', 
      icon: Layers 
    },
    { 
      num: '04', 
      title: 'Auxiliary Facial Geometry & Anthropometric Signals', 
      desc: 'Localizes 18 anatomical facial landmarks to extract continuous geometry: mouth corner elevation, mouth aspect ratio (MAR), eye aspect ratio (EAR), and glabella eyebrow furrow distance. Operates as an auxiliary signal stream without hardcoding overrides.', 
      icon: Scan 
    },
    { 
      num: '05', 
      title: '22-Dimensional Feature Space Assembly', 
      desc: 'Extracts 7 baseline emotion probabilities, 10 LBP histogram bins, edge density, gradient energy, brightness, contrast, and Laplacian sharpness into a unified feature vector.', 
      icon: BarChart3 
    },
    { 
      num: '06', 
      title: 'Feature-Level Supervised Fusion Model', 
      desc: 'Applies StandardScaler normalization and a trained multiclass Logistic Regression fusion model calibrated on the FER2013 dataset.', 
      icon: Cpu 
    },
    { 
      num: '07', 
      title: 'Temporal EMA Smoothing & Quality Validation', 
      desc: 'Applies Exponential Moving Average (EMA, α = 0.50) temporal filtering across a 5-frame buffer to eliminate single-frame prediction jitter and validate optical illumination.', 
      icon: CheckCircle2 
    }
  ];

  const geometricSignals = [
    { signal: 'Mouth corners raised', indication: 'Happy / Positive Valence', icon: '😊', note: 'Curvature bends upward relative to oral fissure center' },
    { signal: 'Mouth corners roughly level', indication: 'Neutral', icon: '😐', note: 'Balanced baseline resting facial tone' },
    { signal: 'Mouth corners lowered', indication: 'Sad / Negative Valence', icon: '🙁', note: 'Downturned corners provide negative valence evidence' },
    { signal: 'Mouth widely open / O-shape', indication: 'Mouth Aperture', icon: '😮', note: 'Aperture alone does NOT determine surprise (speech/yawn)' },
    { signal: 'Eyebrows pulled together/down', indication: 'Angry / Tension', icon: '😠', note: 'Glabella contraction and furrow distance narrowing' },
    { signal: 'Eyes narrowed + brow furrow', indication: 'Angry / Concentration', icon: '😤', note: 'Co-occurrence of eye narrowing and eyebrow depression' },
    { signal: 'Eyebrows raised + eyes open + mouth open', indication: 'Surprise (Strong Co-Occurrence)', icon: '😲', note: 'Multi-cue co-occurrence provides genuine surprise evidence' },
    { signal: 'Mouth stretched horizontally', indication: 'Smile / Expressive Valence', icon: '😁', note: 'Broadened mouth width relative to face bounding box' },
  ];

  return (
    <div className="space-y-14 py-8 max-w-4xl mx-auto text-slate-100">
      {/* Header */}
      <div className="text-center space-y-3">
        <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">
          Architecture Pipeline & Scientific Principles
        </span>
        <h1 className="text-3xl md:text-5xl font-extrabold tracking-tight text-white">
          How Facial Mood Analysis Works
        </h1>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto leading-relaxed">
          A step-by-step walkthrough of the multi-representation computer vision pipeline, 
          combining DIP representation filters, 22D fusion classification, and auxiliary facial geometry.
        </p>
      </div>

      {/* Critical Scientific Note: Face Localization vs Identity Recognition */}
      <div className="bg-slate-900/90 border border-emerald-500/30 rounded-3xl p-6 md:p-8 space-y-3 shadow-2xl relative overflow-hidden">
        <div className="flex items-center gap-2.5 text-emerald-400 font-bold text-sm">
          <ShieldCheck className="w-5 h-5" />
          <span>Core Principle: Face Detection & Localization — NOT Identity Recognition</span>
        </div>
        <p className="text-xs text-slate-300 leading-relaxed">
          Our system <strong>detects and localizes the face region and anatomical landmark points</strong> strictly to extract scale-normalized expression signals. 
          There is <strong>zero identity recognition, person profiling, or biometric template matching</strong>. 
          Every metric is normalized by the face bounding box dimensions to be scale-invariant and privacy-preserving.
        </p>
      </div>

      {/* Pipeline Steps */}
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

      {/* Auxiliary Facial Geometry Principles Table */}
      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-2xl space-y-6">
        <div className="space-y-1">
          <span className="text-xs font-mono uppercase tracking-widest text-cyan-400 font-semibold">
            Auxiliary Geometric Signals
          </span>
          <h2 className="text-xl font-bold text-white">
            Facial Signals & Multi-Cue Co-Occurrence Principles
          </h2>
          <p className="text-xs text-slate-400 leading-relaxed">
            Geometric cues provide auxiliary evidence to explain and validate the machine learning model. 
            However, <strong>one feature should never directly mean one emotion</strong>.
          </p>
        </div>

        {/* Warning callout */}
        <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-2xl flex items-start gap-3 text-xs text-amber-200">
          <AlertCircle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <strong className="text-white block font-mono text-[11px] uppercase mb-1">
              The Co-Occurrence Principle
            </strong>
            <span className="text-slate-300">
              Mouth open = Surprise ❌ (Someone can open their mouth while talking, singing, or yawning).<br />
              <strong>Mouth open + eyes widened + eyebrows raised → Strong Surprise Signal ✅</strong> (Co-occurring multi-feature evidence).
            </span>
          </div>
        </div>

        {/* Signals Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                <th className="py-3 px-3">Facial Signal</th>
                <th className="py-3 px-3">Indication</th>
                <th className="py-3 px-3">Geometric Implementation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {geometricSignals.map((item, idx) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="py-3.5 px-3 text-slate-200 font-semibold flex items-center gap-2">
                    <span className="text-base">{item.icon}</span> {item.signal}
                  </td>
                  <td className="py-3.5 px-3 text-cyan-300 font-bold">{item.indication}</td>
                  <td className="py-3.5 px-3 text-slate-400">{item.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

