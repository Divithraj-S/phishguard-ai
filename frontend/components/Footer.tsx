import React from "react";
import { AlertTriangle, Shield, CheckCircle } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-cyber-border/80 bg-cyber-darkest/90 mt-12 py-8 text-xs text-slate-400">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
        {/* Prominent Disclaimer Notice Card */}
        <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/30 text-amber-200/90 flex items-start space-x-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h4 className="font-semibold text-amber-300 text-xs">
              Important Security Disclaimer
            </h4>
            <p className="text-xs leading-relaxed text-amber-200/80">
              This is an experimental text classifier. A low phishing score does
              not guarantee that an email is safe. Verify suspicious requests
              through trusted out-of-band channels.
            </p>
          </div>
        </div>

        {/* Scope and Technical Capabilities */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-[11px] text-slate-500 font-mono">
          <div>
            <span className="text-slate-300 block mb-1 font-sans font-semibold">
              Text Patterns Only
            </span>
            Evaluates phrasing, urgent call-to-actions, and n-grams from email body text.
          </div>
          <div>
            <span className="text-slate-300 block mb-1 font-sans font-semibold">
              Out of Scope
            </span>
            Does not scan file attachments, embedded QR codes, live web links, or SPF/DKIM headers.
          </div>
          <div>
            <span className="text-slate-300 block mb-1 font-sans font-semibold">
              Privacy First
            </span>
            Input messages are processed in-memory against frozen weights and are never persisted or used for retraining.
          </div>
        </div>

        {/* Copyright & Project Info */}
        <div className="pt-4 border-t border-cyber-border/40 flex flex-col sm:flex-row items-center justify-between text-slate-500">
          <p>© 2026 PhishGuard AI • Reducing False Positives in Email Detection</p>
          <p className="font-mono text-[11px] mt-2 sm:mt-0">
            Deployed with FastAPI &amp; Next.js
          </p>
        </div>
      </div>
    </footer>
  );
}
