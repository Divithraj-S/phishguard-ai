import React from "react";
import { Shield, Sparkles, Filter, CheckCircle2 } from "lucide-react";

export default function Hero() {
  return (
    <section className="text-center pt-8 pb-6 px-4 sm:px-6 lg:px-8 max-w-4xl mx-auto">
      {/* Badge */}
      <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-mono mb-4 shadow-sm">
        <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
        <span>Optimized for Low False Positives (0.68% FPR)</span>
      </div>

      {/* Main Heading */}
      <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
        Detect Phishing Before It{" "}
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500">
          Reaches You
        </span>
      </h1>

      {/* Short Subtitle */}
      <p className="mt-4 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto leading-relaxed">
        Analyze email text patterns and evaluate decision-threshold trade-offs
        to minimize disruptive false alarms while reliably intercepting cyber threats.
      </p>

      {/* Key Guarantees */}
      <div className="mt-6 flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-xs text-slate-400">
        <div className="flex items-center space-x-1.5">
          <CheckCircle2 className="w-4 h-4 text-cyan-400" />
          <span>Stratified 17,500+ Email Corpus</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <Filter className="w-4 h-4 text-cyan-400" />
          <span>Tunable Decision Boundary (0.20 – 0.90)</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <Shield className="w-4 h-4 text-cyan-400" />
          <span>Zero User-Data Retraining</span>
        </div>
      </div>
    </section>
  );
}
