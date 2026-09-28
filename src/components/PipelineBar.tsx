import React from 'react';
import { Camera, Layers, Cpu, BarChart3, Activity, CheckCircle2 } from 'lucide-react';

export const PipelineBar: React.FC = () => {
  const steps = [
    { label: 'RGB Camera', icon: Camera },
    { label: 'Representations', icon: Layers },
    { label: 'Feature Extraction', icon: BarChart3 },
    { label: '22D Fusion', icon: Cpu },
    { label: 'Temporal EMA', icon: Activity },
    { label: 'Estimate', icon: CheckCircle2 }
  ];

  return (
    <div className="w-full bg-slate-900/90 dark:bg-slate-950/90 border border-slate-800 rounded-2xl p-4 backdrop-blur-xl shadow-2xl overflow-x-auto">
      <div className="flex items-center justify-between min-w-[700px] gap-2">
        {steps.map((step, idx) => {
          const Icon = step.icon;
          return (
            <React.Fragment key={idx}>
              <div className="flex items-center gap-2.5 px-3 py-2 bg-slate-800/60 rounded-xl border border-slate-700/50">
                <div className="w-7 h-7 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center shrink-0">
                  <Icon className="w-3.5 h-3.5" />
                </div>
                <div>
                  <span className="text-[10px] font-mono text-slate-400 block uppercase tracking-wider">Step 0{idx+1}</span>
                  <span className="text-xs font-bold text-slate-100">{step.label}</span>
                </div>
              </div>
              {idx < steps.length - 1 && (
                <div className="flex-1 h-px bg-gradient-to-r from-slate-700 via-cyan-500/40 to-slate-750"></div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
