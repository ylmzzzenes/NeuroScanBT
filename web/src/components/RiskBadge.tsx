import { motion } from "framer-motion";
import type { PredictionPayload } from "../lib/api";

const map = {
  low: {
    label: "Düşük risk",
    className:
      "bg-emerald-500/10 text-emerald-300 border-emerald-500/25 ring-emerald-500/20",
  },
  medium: {
    label: "Orta risk",
    className:
      "bg-amber-500/10 text-amber-200 border-amber-500/25 ring-amber-500/20",
  },
  high: {
    label: "Yüksek risk",
    className: "bg-rose-500/10 text-rose-200 border-rose-500/30 ring-rose-500/25",
  },
};

export function RiskBadge({ level }: { level: PredictionPayload["risk_level"] }) {
  const cfg = map[level];
  return (
    <motion.span
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold ring-1 ${cfg.className}`}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-80" />
      {cfg.label}
    </motion.span>
  );
}
