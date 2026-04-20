import { useEffect, useState } from "react";
import { motion } from "framer-motion";

const MESSAGES = [
  "Beyin taraması analiz ediliyor…",
  "Doku kontrastı haritalanıyor…",
  "Kanama olasılığı değerlendiriliyor…",
  "Güven bantları hesaplanıyor…",
  "Dikkat haritası sentezleniyor…",
];

export function AnalyzingOverlay() {
  const [i, setI] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setI((x) => (x + 1) % MESSAGES.length), 1600);
    return () => clearInterval(t);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-neuro-void/85 backdrop-blur-md"
    >
      <div className="relative mx-4 max-w-md text-center">
        <motion.div
          className="mx-auto h-28 w-28 rounded-full"
          style={{
            background:
              "conic-gradient(from 0deg, rgba(45,212,191,0.4), transparent, rgba(167,139,250,0.35), transparent)",
          }}
          animate={{ rotate: 360 }}
          transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
        />
        <motion.div
          className="absolute left-1/2 top-1/2 h-16 w-16 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white/10 bg-neuro-deep/90 shadow-glass"
          animate={{ scale: [1, 1.06, 1] }}
          transition={{ repeat: Infinity, duration: 2.2 }}
        />
        <motion.p
          key={i}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8 font-display text-sm font-medium text-neuro-ice"
        >
          {MESSAGES[i]}
        </motion.p>
        <p className="mt-2 text-xs text-slate-500">
          Sinir ağı çıkarımı sürüyor — lütfen bekleyin
        </p>
      </div>
    </motion.div>
  );
}
