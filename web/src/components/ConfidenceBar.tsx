import { motion } from "framer-motion";

export function ConfidenceBar({
  hemorrhagePct,
  noPct,
}: {
  hemorrhagePct: number;
  noPct: number;
}) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-[11px] uppercase tracking-wide text-slate-500">
        <span>Kanama sinyali</span>
        <span>Stabil doku</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-white/[0.06] ring-1 ring-white/[0.06]">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-rose-500/90 via-neuro-warn/80 to-neuro-cyan"
          initial={{ width: 0 }}
          animate={{ width: `${hemorrhagePct}%` }}
          transition={{ type: "spring", stiffness: 80, damping: 20 }}
        />
      </div>
      <div className="flex justify-between text-xs tabular-nums text-slate-400">
        <span>{hemorrhagePct.toFixed(1)}%</span>
        <span>{noPct.toFixed(1)}%</span>
      </div>
    </div>
  );
}
