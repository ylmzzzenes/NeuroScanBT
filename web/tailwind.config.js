/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"DM Sans"',
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "sans-serif",
        ],
        display: ['"Outfit"', "system-ui", "sans-serif"],
      },
      colors: {
        neuro: {
          void: "#05070c",
          deep: "#0a0f18",
          panel: "#0f1624",
          glass: "rgba(255,255,255,0.04)",
          stroke: "rgba(120, 200, 255, 0.12)",
          cyan: "#2dd4bf",
          ice: "#67e8f9",
          violet: "#a78bfa",
          rose: "#fb7185",
          warn: "#fbbf24",
        },
      },
      backgroundImage: {
        "grid-fade":
          "linear-gradient(to bottom, rgba(5,7,12,0.2), rgba(5,7,12,1)), radial-gradient(ellipse 80% 50% at 50% -20%, rgba(45,212,191,0.15), transparent)",
        "glow-conic":
          "conic-gradient(from 180deg at 50% 50%, rgba(45,212,191,0.15), transparent 40%, rgba(167,139,250,0.12), transparent 75%)",
      },
      boxShadow: {
        glass: "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06)",
        neon: "0 0 24px rgba(45,212,191,0.25)",
      },
      animation: {
        shimmer: "shimmer 2.2s ease-in-out infinite",
        "pulse-soft": "pulse-soft 2.5s ease-in-out infinite",
      },
      keyframes: {
        shimmer: {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        "pulse-soft": {
          "0%, 100%": { transform: "scale(1)", opacity: "0.85" },
          "50%": { transform: "scale(1.03)", opacity: "1" },
        },
      },
    },
  },
  plugins: [],
};
