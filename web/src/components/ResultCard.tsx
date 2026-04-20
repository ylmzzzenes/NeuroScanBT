import { motion } from "framer-motion";
import type { PredictionPayload } from "../lib/api";
import { GUVEN_ARALIGI_ACIKLAMA, kararMetniEtiket } from "../lib/tr";
import { ConfidenceBar } from "./ConfidenceBar";
import { HeatmapCard } from "./HeatmapCard";
import { RadialMeter } from "./RadialMeter";
import { RiskBadge } from "./RiskBadge";

export function ResultCard({
  data,
  title,
}: {
  data: PredictionPayload;
  title: string;
}) {
  const hem = data.hemorrhage_probability * 100;
  const no = data.no_hemorrhage_probability * 100;
  const positive = data.label === "hemorrhage";

  return (
    <motion.article
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel flex flex-col gap-5 p-5 sm:p-6"
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-neuro-cyan/90">
            {title}
            {data.model_display_name ? (
              <span className="ml-2 text-neuro-ice/80">· {data.model_display_name}</span>
            ) : null}
          </p>
          <h2
            className={`mt-1 font-display text-2xl font-bold leading-tight sm:text-3xl ${
              positive ? "text-rose-200" : "text-neuro-ice"
            }`}
          >
            {kararMetniEtiket(data.label)}
          </h2>
        </div>
        <RiskBadge level={data.risk_level} />
      </header>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 sm:items-start">
        <RadialMeter
          value={data.confidence_percent}
          label="Model güveni"
          sub="üst sınıf"
        />
        <div className="space-y-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Kanama olasılığı güven aralığı
          </p>
          <p className="font-display text-lg font-semibold tabular-nums text-white">
            {(data.confidence_interval.low * 100).toFixed(1)}% —{" "}
            {(data.confidence_interval.high * 100).toFixed(1)}%
          </p>
          <p className="text-[10px] leading-relaxed text-slate-500">
            {GUVEN_ARALIGI_ACIKLAMA}
          </p>
        </div>
      </div>

      <ConfidenceBar hemorrhagePct={hem} noPct={no} />

      <HeatmapCard data={data} />
    </motion.article>
  );
}
