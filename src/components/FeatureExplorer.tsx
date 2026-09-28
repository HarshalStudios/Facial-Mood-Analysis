import React from 'react';
import { BarChart3, Layers, Sparkles, Cpu } from 'lucide-react';

export const FeatureExplorer: React.FC = () => {
  const featureDefinitions = [
    { range: '0 - 6', name: 'Emotion Probabilities', source: 'FER2013 Linear Model / DeepFace', definition: 'Seven supervised probability scores reordered to canonical emotion order', rangeVal: '0.0 – 1.0' },
    { range: '7 - 16', name: 'LBP Histogram Bins (10 bins)', source: 'Local Binary Pattern (P=8, R=1)', definition: 'Normalized histogram of uniform LBP codes capturing texture gradients', rangeVal: '0.0 – 1.0 (~sum 1)' },
    { range: '17', name: 'Edge Density', source: 'Canny Edge Detector', definition: 'Count of non-zero edge pixels divided by total frame/ROI pixels', rangeVal: '0.0 – 1.0' },
    { range: '18', name: 'Gradient Energy', source: 'Sobel Gradients (Gx² + Gy²)', definition: 'Mean squared gradient energy on normalized grayscale representation', rangeVal: 'Positive float' },
    { range: '19', name: 'Brightness', source: 'Grayscale Statistics', definition: 'Arithmetic mean pixel intensity of the grayscale face crop', rangeVal: '0.0 – 255.0' },
    { range: '20', name: 'Contrast', source: 'Grayscale Statistics', definition: 'Standard deviation of pixel intensities across face ROI', rangeVal: '≥ 0.0' },
    { range: '21', name: 'Sharpness', source: 'Laplacian Variance', definition: 'Variance of the Laplacian filter response indicating blur/sharpness', rangeVal: '≥ 0.0' }
  ];

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <div className="max-w-3xl">
          <h2 className="text-xl font-bold text-slate-900 mb-2">22-Dimensional Multi-Representation Feature Vector</h2>
          <p className="text-sm text-slate-600">
            The facial mood analysis engine constructs a robust 22-dimensional feature vector per face crop by fusing supervised emotion probability estimates with handcrafted digital image processing (DIP) representations.
          </p>
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-6 py-4 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <h3 className="text-xs font-mono uppercase tracking-wider text-slate-500 font-semibold">Feature Vector Schema Index</h3>
          <span className="text-xs font-mono text-cyan-700 bg-cyan-50 px-2.5 py-1 rounded border border-cyan-200">Total Dimensions: 22</span>
        </div>
        <div className="divide-y divide-slate-200">
          {featureDefinitions.map((item, idx) => (
            <div key={idx} className="px-6 py-4 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-slate-50/50 transition-colors">
              <div className="space-y-1">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs font-bold px-2 py-1 bg-slate-100 text-slate-700 rounded border border-slate-200">
                    Indices {item.range}
                  </span>
                  <h4 className="text-sm font-bold text-slate-900">{item.name}</h4>
                </div>
                <p className="text-xs text-slate-600 pl-1">{item.definition}</p>
              </div>
              <div className="flex items-center gap-4 text-right">
                <div>
                  <span className="text-[10px] font-mono text-slate-400 block uppercase">Source</span>
                  <span className="text-xs font-medium text-slate-800">{item.source}</span>
                </div>
                <div className="hidden lg:block text-left pl-4 border-l border-slate-200">
                  <span className="text-[10px] font-mono text-slate-400 block uppercase">Range</span>
                  <span className="text-xs font-mono font-bold text-cyan-700">{item.rangeVal}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
