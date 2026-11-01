"use client";

import React, { useState } from "react";
import {
  FileText,
  Sliders,
  Send,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Info,
  ShieldAlert,
  ShieldCheck,
  Loader2,
  CheckCircle2,
  HelpCircle,
} from "lucide-react";
import { predictEmail, PredictionResponse } from "@/lib/api";

// Realistic sample emails for interactive testing
export const SAMPLES = [
  {
    id: "legit",
    label: "Legitimate Message",
    tag: "Safe Sample",
    badgeColor: "text-cyan-400 border-cyan-500/30 bg-cyan-950/40",
    text: `Hi Sarah,

Here are the minutes and action items from today's sprint planning session:
- Finalize Q3 cybersecurity roadmap by Thursday.
- Review API latency metrics with the infrastructure team.
- Update internal documentation for new team members.

Let me know if you need any adjustments before the client presentation.

Best regards,
David Miller | Senior Operations Manager`,
  },
  {
    id: "phish",
    label: "Suspicious Urgent Alert",
    tag: "Phishing Sample",
    badgeColor: "text-amber-400 border-amber-500/30 bg-amber-950/40",
    text: `URGENT SECURITY ALERT: Unauthorized Access Detected!

Your corporate single-sign-on session was initiated from an unrecognized device in Moscow, Russia. 

To safeguard your organization's mailbox, you are required to verify your employee credentials within the next 60 minutes. Failure to confirm your identity will result in permanent account suspension.

Click the secure link below to verify identity now:
http://corp-auth-verify-security-update.com/login?id=89234

IT Security Helpdesk`,
  },
  {
    id: "neutral",
    label: "Vendor Invoice Notice",
    tag: "Neutral / Borderline",
    badgeColor: "text-blue-400 border-blue-500/30 bg-blue-950/40",
    text: `Dear Accounts Payable,

Please find attached invoice #INV-2026-8819 for cloud infrastructure support services delivered last month.

Payment terms are Net-30. If you have already remitted the payment via wire transfer, please disregard this automated notification.

For billing inquiries or tax exemptions, reply directly to this email.

Thank you,
Apex Cloud Solutions Billing Department`,
  },
];

export default function AnalysisPanel() {
  const [emailText, setEmailText] = useState<string>("");
  const [threshold, setThreshold] = useState<number>(0.60);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  const characterCount = emailText.length;
  const wordCount = emailText.trim() ? emailText.trim().split(/\s+/).length : 0;

  const handleThresholdChange = (newThreshold: number) => {
    setThreshold(newThreshold);
    // If an email has already been analyzed, immediately re-evaluate via the real backend
    // to show how the decision label changes with the new threshold
    if (result && emailText.trim() && !loading) {
      predictEmail({
        email_text: emailText.trim(),
        threshold: newThreshold,
      })
        .then((data) => setResult(data))
        .catch((err) => {
          setError(err.message || "Failed to update prediction for new threshold.");
        });
    }
  };

  const handleSelectSample = (text: string) => {
    setEmailText(text);
    setError(null);
    setResult(null);
  };

  const handleClear = () => {
    setEmailText("");
    setError(null);
    setResult(null);
  };

  const handleAnalyze = async () => {
    const trimmed = emailText.trim();
    if (!trimmed) {
      setError("Please paste or type an email message to analyze.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Calls the real FastAPI backend endpoint POST /api/predict
      const data = await predictEmail({
        email_text: trimmed,
        threshold: threshold,
      });
      setResult(data);
    } catch (err: any) {
      setError(
        err.message ||
          "Could not connect to the PhishGuard API. Please ensure the FastAPI backend is running on http://127.0.0.1:8000."
      );
    } finally {
      setLoading(false);
    }
  };

  // Derived values for visualization
  const pPhishPct = result ? result.phishing_probability * 100 : 0;
  const pSafePct = result ? result.safe_probability * 100 : 0;
  const thresholdPct = result ? result.decision_threshold * 100 : threshold * 100;
  const isPhishing = result?.prediction_label === "Phishing Email";

  return (
    <div className="w-full max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Input and Threshold Controls (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Main Input Card */}
          <div className="bg-cyber-card/90 rounded-2xl border border-cyber-border shadow-xl backdrop-blur-sm p-5 sm:p-6 transition-all hover:border-cyber-borderLight">
            {/* Header & Sample Buttons */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileText className="w-5 h-5 text-cyan-400" />
                  <h2 className="text-base font-semibold text-white">
                    Email Message Content
                  </h2>
                </div>
                {emailText && (
                  <button
                    onClick={handleClear}
                    disabled={loading}
                    className="flex items-center space-x-1 text-xs text-slate-400 hover:text-slate-200 transition-colors disabled:opacity-50"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Clear text</span>
                  </button>
                )}
              </div>

              {/* Sample Buttons */}
              <div>
                <p className="text-xs text-slate-400 mb-2 font-mono">
                  Quick-load realistic sample:
                </p>
                <div className="flex flex-wrap gap-2">
                  {SAMPLES.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => handleSelectSample(s.text)}
                      disabled={loading}
                      className={`text-xs px-3 py-1.5 rounded-lg border font-medium transition-all duration-150 hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50 ${s.badgeColor}`}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Textarea */}
            <div className="mt-4">
              <textarea
                value={emailText}
                onChange={(e) => {
                  setEmailText(e.target.value);
                  if (error) setError(null);
                }}
                disabled={loading}
                rows={9}
                placeholder="Paste the full email text here (subject line and message body)..."
                className="w-full px-4 py-3 rounded-xl bg-cyber-darkest border border-cyber-border/80 text-slate-100 placeholder-slate-500 text-sm font-sans focus:outline-none focus:border-cyan-500/80 focus:ring-1 focus:ring-cyan-500/50 transition-all resize-none disabled:opacity-60"
              />
              <div className="flex items-center justify-between mt-2 text-[11px] text-slate-500 font-mono">
                <span>
                  Words: <strong className="text-slate-400">{wordCount}</strong> | Characters:{" "}
                  <strong className="text-slate-400">{characterCount}</strong>
                </span>
                <span>Language: English (TF-IDF 10k Vocab)</span>
              </div>
            </div>

            {/* Threshold Slider Card */}
            <div className="mt-6 pt-5 border-t border-cyber-border/60">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  <label htmlFor="threshold-slider" className="text-xs font-semibold text-slate-200">
                    Decision Threshold:
                  </label>
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                    {threshold.toFixed(2)}
                  </span>
                </div>
                <span className="text-[11px] text-slate-400 font-mono">
                  {(threshold * 100).toFixed(0)}% confidence required
                </span>
              </div>

              {/* Slider Input */}
              <input
                id="threshold-slider"
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={threshold}
                disabled={loading}
                onChange={(e) => handleThresholdChange(parseFloat(e.target.value))}
                className="w-full h-2 bg-cyber-darkest rounded-lg appearance-none cursor-pointer accent-cyan-400 focus:outline-none disabled:opacity-50"
              />

              {/* Preset Buttons */}
              <div className="mt-3 grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => handleThresholdChange(0.40)}
                  disabled={loading}
                  className={`text-[11px] py-1.5 px-2 rounded-lg border text-center transition-all ${
                    threshold === 0.40
                      ? "border-cyan-400 bg-cyan-500/15 text-cyan-200 font-semibold"
                      : "border-cyber-border bg-cyber-darkest text-slate-400 hover:text-slate-200"
                  }`}
                >
                  More sensitive (0.40)
                </button>
                <button
                  type="button"
                  onClick={() => handleThresholdChange(0.50)}
                  disabled={loading}
                  className={`text-[11px] py-1.5 px-2 rounded-lg border text-center transition-all ${
                    threshold === 0.50
                      ? "border-cyan-400 bg-cyan-500/15 text-cyan-200 font-semibold"
                      : "border-cyber-border bg-cyber-darkest text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Balanced (0.50)
                </button>
                <button
                  type="button"
                  onClick={() => handleThresholdChange(0.60)}
                  disabled={loading}
                  className={`text-[11px] py-1.5 px-2 rounded-lg border text-center transition-all ${
                    threshold === 0.60
                      ? "border-cyan-400 bg-cyan-500/15 text-cyan-200 font-semibold"
                      : "border-cyber-border bg-cyber-darkest text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Fewer false alarms (0.60)
                </button>
              </div>
            </div>

            {/* Error Message Banner */}
            {error && (
              <div className="mt-4 p-3 rounded-xl bg-red-950/30 border border-red-500/40 text-red-300 text-xs flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="space-y-0.5">
                  <p className="font-semibold text-red-200">Evaluation Notice</p>
                  <p>{error}</p>
                </div>
              </div>
            )}

            {/* Analyze Button */}
            <div className="mt-6">
              <button
                type="button"
                onClick={handleAnalyze}
                disabled={loading || !emailText.trim()}
                className="w-full flex items-center justify-center space-x-2 py-3.5 px-6 rounded-xl font-semibold text-sm transition-all duration-150 shadow-cyan-glow bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing Vocabulary &amp; Phrasing...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span>Analyze Email Pattern</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Results & Educational Explanation (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Results Dynamic Card */}
          <div className="bg-cyber-card/90 rounded-2xl border border-cyber-border shadow-xl backdrop-blur-sm p-5 sm:p-6 transition-all">
            <div className="flex items-center justify-between border-b border-cyber-border/60 pb-3">
              <div className="flex items-center space-x-2">
                <Sparkles className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-semibold text-white">
                  Detection Assessment
                </h3>
              </div>
              <span
                className={`text-[10px] uppercase font-mono px-2 py-0.5 rounded border ${
                  result
                    ? isPhishing
                      ? "bg-amber-950/40 text-amber-300 border-amber-500/40"
                      : "bg-cyan-950/40 text-cyan-300 border-cyan-500/40"
                    : "bg-slate-800 text-slate-400 border-slate-700"
                }`}
              >
                {result ? "Evaluation Complete" : "Awaiting Analysis"}
              </span>
            </div>

            {/* Placeholder state when no prediction has run */}
            {!result && !loading && (
              <div className="py-10 text-center space-y-3">
                <div className="w-12 h-12 rounded-full bg-cyber-darkest border border-cyber-border flex items-center justify-center mx-auto text-slate-500">
                  <FileText className="w-6 h-6" />
                </div>
                <p className="text-sm text-slate-300 font-medium">
                  No email analyzed yet
                </p>
                <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                  Paste an email message on the left or click a sample button, then
                  click &quot;Analyze Email Pattern&quot; to inspect model probabilities.
                </p>
              </div>
            )}

            {/* Loading state animation */}
            {loading && (
              <div className="py-12 text-center space-y-4">
                <div className="w-12 h-12 rounded-full bg-cyan-950/40 border border-cyan-500/40 flex items-center justify-center mx-auto text-cyan-400 animate-spin">
                  <Loader2 className="w-6 h-6" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm text-slate-200 font-medium">
                    Vectorizing N-Grams &amp; Scoring Risk...
                  </p>
                  <p className="text-xs text-slate-500">
                    Evaluating text through Logistic Regression weights
                  </p>
                </div>
              </div>
            )}

            {/* Real Prediction Results View */}
            {result && !loading && (
              <div className="py-4 space-y-5">
                {/* Category Badge Card */}
                <div
                  className={`p-4 rounded-xl border flex items-start space-x-3.5 transition-all ${
                    isPhishing
                      ? "bg-amber-950/30 border-amber-500/40 text-amber-200"
                      : "bg-cyan-950/30 border-cyan-500/40 text-cyan-200"
                  }`}
                >
                  <div className="mt-0.5">
                    {isPhishing ? (
                      <ShieldAlert className="w-6 h-6 text-amber-400" />
                    ) : (
                      <ShieldCheck className="w-6 h-6 text-cyan-400" />
                    )}
                  </div>
                  <div>
                    <span className="text-[10px] font-mono font-semibold uppercase tracking-wider block opacity-75">
                      {isPhishing ? "Threat Detected" : "Benign Pattern"}
                    </span>
                    <h4 className="text-lg font-bold text-white tracking-tight">
                      {result.prediction_label}
                    </h4>
                    <p className="text-xs text-slate-300 mt-1 leading-relaxed">
                      {isPhishing
                        ? `Estimated phishing risk (${pPhishPct.toFixed(1)}%) meets or exceeds the active decision threshold (${thresholdPct.toFixed(0)}%).`
                        : `Estimated phishing risk (${pPhishPct.toFixed(1)}%) is below the active decision threshold (${thresholdPct.toFixed(0)}%).`}
                    </p>
                  </div>
                </div>

                {/* Quantitative Metrics Grid */}
                <div className="grid grid-cols-2 gap-3 font-mono">
                  <div className="p-3 rounded-xl bg-cyber-darkest border border-cyber-border space-y-1">
                    <span className="text-[10px] uppercase text-slate-400 block font-sans">
                      Phishing Risk Score
                    </span>
                    <span className="text-xl font-bold text-white">
                      {pPhishPct.toFixed(2)}%
                    </span>
                    <span className="text-[11px] block text-slate-400">
                      {pPhishPct >= thresholdPct ? (
                        <span className="text-amber-400">
                          +{(pPhishPct - thresholdPct).toFixed(1)}% vs threshold
                        </span>
                      ) : (
                        <span className="text-slate-400">
                          {(pPhishPct - thresholdPct).toFixed(1)}% vs threshold
                        </span>
                      )}
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-cyber-darkest border border-cyber-border space-y-1">
                    <span className="text-[10px] uppercase text-slate-400 block font-sans">
                      Safe Probability
                    </span>
                    <span className="text-xl font-bold text-slate-200">
                      {pSafePct.toFixed(2)}%
                    </span>
                    <span className="text-[11px] block text-slate-500">
                      P(Safe) = 1 - P(Phish)
                    </span>
                  </div>
                </div>

                {/* Probability Spectrum Visualization */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-300 font-mono">
                    <span>Probability Spectrum</span>
                    <span className="text-cyan-400">Score: {pPhishPct.toFixed(1)}%</span>
                  </div>

                  {/* Relative Track */}
                  <div className="relative h-4 rounded-full bg-cyber-darkest border border-cyber-border overflow-hidden">
                    {/* Fill Bar */}
                    <div
                      style={{ width: `${Math.min(Math.max(pPhishPct, 0), 100)}%` }}
                      className={`h-full transition-all duration-500 ${
                        isPhishing
                          ? "bg-gradient-to-r from-amber-500 to-red-500"
                          : "bg-gradient-to-r from-cyan-500 to-blue-500"
                      }`}
                    />

                    {/* Threshold Marker Indicator */}
                    <div
                      style={{ left: `${Math.min(Math.max(thresholdPct, 0), 100)}%` }}
                      title={`Decision Threshold: ${thresholdPct.toFixed(0)}%`}
                      className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
                    />
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>0% (Benign)</span>
                    <span className="text-slate-300">
                      Threshold: {thresholdPct.toFixed(0)}%
                    </span>
                    <span>100% (High Risk)</span>
                  </div>
                </div>

                {/* Decision Rule Pedagogical Explanation */}
                <div className="p-3.5 rounded-xl bg-cyber-darkest/90 border border-cyber-border text-xs text-slate-300 space-y-2">
                  <div className="flex items-center space-x-1.5 font-semibold text-slate-200">
                    <HelpCircle className="w-4 h-4 text-cyan-400" />
                    <span>Decision Explanation:</span>
                  </div>
                  <p className="leading-relaxed text-slate-300">
                    {isPhishing ? (
                      <>
                        Because the estimated phishing probability (<strong>{pPhishPct.toFixed(2)}%</strong>) is <strong>greater than or equal to</strong> your decision threshold (<strong>{thresholdPct.toFixed(0)}%</strong>), the model classified this message as <strong>Phishing Email</strong>.
                      </>
                    ) : (
                      <>
                        Because the estimated phishing probability (<strong>{pPhishPct.toFixed(2)}%</strong>) is <strong>lower than</strong> your decision threshold (<strong>{thresholdPct.toFixed(0)}%</strong>), the model classified this message as <strong>Safe Email</strong>.
                      </>
                    )}
                  </p>
                  <p className="text-[11px] text-slate-400 border-t border-cyber-border/60 pt-2 leading-relaxed">
                    💡 <strong>Threshold Impact:</strong> Under our Phase 8 audit, setting the threshold to 0.60 reduced false alarms by 42.3% while maintaining over 93% threat recall. Adjusting the slider dynamically alters the standard of evidence required to trigger a flag.
                  </p>
                </div>

                {/* Model Disclaimer Direct from API */}
                <p className="text-[11px] text-slate-500 italic leading-relaxed border-t border-cyber-border/40 pt-2">
                  🔒 {result.security_disclaimer}
                </p>
              </div>
            )}

            {/* Threshold Explanation Box when no result is present */}
            {!result && !loading && (
              <div className="mt-4 p-3.5 rounded-xl bg-cyber-darkest/70 border border-cyber-border/80 text-xs text-slate-400 space-y-1.5">
                <div className="flex items-center space-x-1.5 text-slate-300 font-medium">
                  <Info className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                  <span>Decision Logic:</span>
                </div>
                <p className="leading-relaxed">
                  An email is classified as <strong className="text-amber-300">Phishing</strong> when the calculated risk meets or exceeds your threshold ({threshold.toFixed(2)}). Higher thresholds demand stronger evidence, cutting false alarms by up to 42%.
                </p>
              </div>
            )}
          </div>

          {/* Model Specification Card */}
          <div className="bg-cyber-card/50 rounded-2xl border border-cyber-border/60 p-4 text-xs text-slate-400 space-y-2 font-mono">
            <div className="flex justify-between">
              <span>Model:</span>
              <span className="text-slate-200">Logistic Regression</span>
            </div>
            <div className="flex justify-between">
              <span>Feature Space:</span>
              <span className="text-slate-200">10,000 TF-IDF N-Grams</span>
            </div>
            <div className="flex justify-between">
              <span>Audit FPR @ 0.60:</span>
              <span className="text-cyan-400 font-semibold">0.68% (Held-Out Test)</span>
            </div>
            <div className="flex justify-between">
              <span>Audit Recall @ 0.60:</span>
              <span className="text-cyan-400 font-semibold">93.65% (Held-Out Test)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
