import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cyber: {
          darkest: "#070b14",
          bg: "#0b1120",
          card: "#0f172a",
          cardHover: "#162036",
          border: "#1e293b",
          borderLight: "#334155",
          cyan: "#06b6d4",
          cyanGlow: "#22d3ee",
          blue: "#3b82f6",
          blueGlow: "#60a5fa",
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "cyber-grid": "linear-gradient(to right, rgba(30, 41, 59, 0.2) 1px, transparent 1px), linear-gradient(to bottom, rgba(30, 41, 59, 0.2) 1px, transparent 1px)",
      },
      boxShadow: {
        "cyan-glow": "0 0 25px -5px rgba(6, 182, 212, 0.25)",
        "blue-glow": "0 0 25px -5px rgba(59, 130, 246, 0.25)",
      },
    },
  },
  plugins: [],
};

export default config;
