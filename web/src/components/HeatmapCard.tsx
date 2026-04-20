import { motion } from "framer-motion";
import type { PredictionPayload } from "../lib/api";

export function HeatmapCard({ data }: { data: PredictionPayload }) {
  const imgB64 = data.heatmap_png_base64;
  if (!imgB64) {
    return (
      <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 text-center text-xs text-slate-500">
        Bu çalıştırmada dikkat haritası oluşturulamadı.
      </div>
    );
  }
  const src = `data:image/png;base64,${imgB64}`;
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="overflow-hidden rounded-xl border border-neuro-cyan/20 bg-black/20 ring-1 ring-white/[0.05]"
    >
      <div className="border-b border-white/[0.06] px-3 py-2">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-neuro-ice">
          Gradyan dikkat haritası
        </p>
        <p className="text-[10px] text-slate-500">Sınıfa koşullu saliency katmanı</p>
      </div>
      <img src={src} alt="Saliency ısı haritası" className="w-full object-contain" />
    </motion.div>
  );
}
