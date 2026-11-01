"use client";

import React, { useEffect, useState } from "react";
import { ShieldCheck, Lock, RefreshCw, AlertCircle } from "lucide-react";
import { checkBackendHealth, HealthResponse } from "@/lib/api";

export default function Navbar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const [checking, setChecking] = useState<boolean>(true);

  const fetchHealth = async () => {
    setChecking(true);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
      setIsOnline(data.status === "healthy" && data.model_loaded);
    } catch {
      setIsOnline(false);
      setHealth(null);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    // Periodically re-check health every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="w-full border-b border-cyber-border/80 bg-cyber-darkest/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Logo and Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 p-[1px] shadow-cyan-glow flex items-center justify-center">
            <div className="w-full h-full bg-cyber-darkest rounded-[11px] flex items-center justify-center">
              <ShieldCheck className="w-6 h-6 text-cyan-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white font-mono">
                Phish<span className="text-cyan-400">Guard</span> AI
              </span>
              <span className="text-[10px] uppercase font-semibold tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                v1.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">
              Machine Learning Threat Prevention
            </p>
          </div>
        </div>

        {/* Operational Status Pill connected to GET /api/health */}
        <div className="flex items-center space-x-3 sm:space-x-4">
          <div
            title={
              isOnline
                ? `FastAPI backend online • Model: ${health?.model_name ?? "Logistic Regression"}`
                : "Cannot reach FastAPI backend at /api/health. Please ensure Uvicorn is running."
            }
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs transition-colors ${
              isOnline
                ? "bg-cyber-card border-cyber-border text-slate-300"
                : isOnline === false
                ? "bg-amber-950/30 border-amber-500/40 text-amber-300"
                : "bg-cyber-card border-cyber-border text-slate-400"
            }`}
          >
            {checking ? (
              <RefreshCw className="w-3 h-3 text-cyan-400 animate-spin" />
            ) : isOnline ? (
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            ) : (
              <span className="w-2 h-2 rounded-full bg-amber-400" />
            )}

            <span className="font-mono text-[11px] hidden md:inline">API:</span>
            <span className="font-medium text-[11px] sm:text-xs">
              {isOnline
                ? "Engine Ready"
                : isOnline === false
                ? "Backend Offline"
                : "Checking..."}
            </span>

            <button
              onClick={fetchHealth}
              disabled={checking}
              aria-label="Refresh backend status"
              className="text-slate-500 hover:text-slate-300 transition-colors ml-1"
            >
              <RefreshCw className={`w-3 h-3 ${checking ? "animate-spin" : ""}`} />
            </button>
          </div>

          <div className="hidden lg:flex items-center space-x-1.5 text-xs text-slate-400 font-mono">
            <Lock className="w-3.5 h-3.5 text-slate-500" />
            <span>Frozen Weights (No Retraining)</span>
          </div>
        </div>
      </div>
    </header>
  );
}
