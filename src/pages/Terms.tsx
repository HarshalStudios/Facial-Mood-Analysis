import React from 'react';

export const Terms: React.FC = () => {
  return (
    <div className="space-y-8 py-8 max-w-4xl mx-auto text-slate-100">
      <div className="bg-slate-950 p-10 rounded-3xl border border-slate-800 shadow-2xl space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-xs font-mono text-cyan-400">
          Terms of Use
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-white">Terms & Conditions</h1>
        <p className="text-xs text-slate-400">Last updated: September 28, 2026</p>
      </div>

      <div className="bg-slate-900/80 backdrop-blur-xl rounded-3xl border border-slate-800 p-8 shadow-xl space-y-6 text-sm text-slate-300 leading-relaxed">
        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">1. Acceptance of Terms</h3>
          <p>
            By accessing and using the Facial Mood Analysis web application, you agree to comply with and be bound by these Terms of Use.
          </p>
        </section>

        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">2. Research & Educational Purpose</h3>
          <p>
            This software is designed strictly for research, educational, and computer vision demonstration purposes. It does not provide psychological, medical, or psychiatric diagnostic services.
          </p>
        </section>

        <section className="space-y-2">
          <h3 className="text-base font-bold text-white">3. Limitation of Liability</h3>
          <p>
            The creators and maintainers of this application shall not be held liable for any direct, indirect, incidental, or consequential damages arising from the use or inability to use this software.
          </p>
        </section>
      </div>
    </div>
  );
};
