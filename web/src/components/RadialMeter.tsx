import { motion } from "framer-motion";

export function RadialMeter({
  value,
  label,
  sub,
}: {
  value: number;
  label: string;
  sub?: string;
}) {
  const v = Math.min(100, Math.max(0, value));
  const r = 52;
  const c = 2 * Math.PI * r;
  const offset = c - (v / 100) * c;

  return (
    <div className="flex flex-col items-center">
      <div className="relative h-32 w-32">
        <svg className="-rotate-90" viewBox="0 0 120 120">
          <circle
            cx="60"
            cy="60"
            r={r}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="8"
          />
          <motion.circle
            cx="60"
            cy="60"
            r={r}
            fill="none"
            stroke="url(#gradConf)"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={c}
            initial={{ strokeDashoffset: c }}
            animate={{ strokeDashoffset: offset }}
            transition={{ type: "spring", stiffness: 60, damping: 18 }}
          />
          <defs>
            <linearGradient id="gradConf" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#2dd4bf" />
              <stop offset="100%" stopColor="#67e8f9" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-xl font-bold text-white tabular-nums">
            {v.toFixed(1)}
            <span className="text-sm font-medium text-slate-400">%</span>
          </span>
          {sub && (
            <span className="text-[10px] uppercase tracking-wider text-slate-500">
              {sub}
            </span>
          )}
        </div>
      </div>
      <p className="mt-1 text-center text-[11px] font-medium text-slate-400">
        {label}
      </p>
    </div>
  );
}
