import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PhishGuard AI — Intelligent Phishing Email Detection",
  description:
    "Protect inboxes with machine-learning email text analysis and decision threshold optimization designed to reduce disruptive false positives.",
  keywords: [
    "phishing detection",
    "cybersecurity",
    "machine learning",
    "false positive reduction",
    "email security",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-cyber-darkest text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        <div className="relative min-h-screen overflow-x-hidden flex flex-col">
          {/* Subtle cyber background grid & ambient light glow */}
          <div className="fixed inset-0 pointer-events-none bg-cyber-grid bg-[size:40px_40px] opacity-[0.12] z-0" />
          <div className="fixed top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[400px] bg-gradient-to-b from-cyan-500/10 via-blue-600/5 to-transparent blur-3xl pointer-events-none z-0" />

          <div className="relative z-10 flex-1 flex flex-col">{children}</div>
        </div>
      </body>
    </html>
  );
}
