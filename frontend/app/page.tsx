import React from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import AnalysisPanel from "@/components/AnalysisPanel";
import Footer from "@/components/Footer";

export default function HomePage() {
  return (
    <div className="min-h-screen flex flex-col justify-between">
      <div>
        <Navbar />
        <main>
          <Hero />
          <AnalysisPanel />
        </main>
      </div>
      <Footer />
    </div>
  );
}
