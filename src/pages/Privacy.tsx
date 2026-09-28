import React from 'react';
import { Shield } from 'lucide-react';

export const Privacy: React.FC = () => {
  return (
    <div className="space-y-8 py-8 max-w-4xl mx-auto text-slate-100">
      <div className="bg-slate-950 p-10 rounded-3xl border border-slate-800 shadow-2xl space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-xs font-mono text-cyan-400">
          <Shield className="w-3.5 h-3.5" /> Privacy Architecture
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white">Privacy Policy</h1>
        <p className="text-xs text-slate-400">Last updated: September 28, 2026</p>
      </div>

      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-6 text-sm text-slate-300 leading-relaxed">
        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">1. Camera Permissions & Video Streams</h3>
          <p>
            The Facial Mood Analysis system accesses your webcam only upon your explicit user action when clicking "Enable Camera" or "Start Camera". Video streams are processed locally in your browser and transmitted securely to the backend for frame feature extraction. No video feeds are permanently stored on remote servers.
          </p>
        </section>

        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">2. Local Storage</h3>
          <p>
            Analysis history, timestamps, and predicted emotion scores are stored locally in your browser's localStorage for your convenience. You can clear this history at any time using the "Clear History" button in the History archive.
          </p>
        </section>

        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">3. Third-Party Sharing</h3>
          <p>
            We do not share any personal data, camera frames, or facial feature representations with third parties. All processing occurs within the secure execution environment.
          </p>
        </section>
      </div>
    </div>
  );
};
