import { motion } from "framer-motion";
import type { ModelId } from "../lib/api";

export type RunMode = ModelId | "compare";

const options: {
  id: RunMode;
  title: string;
  subtitle: string;
  accent: string;
}[] = [
  {
    id: "pretrained",
    title: "Önceden eğitilmiş",
    subtitle: "ResNet-18 · transfer öğrenme",
    accent: "from-neuro-cyan/20 to-transparent",
  },
  {
    id: "custom",
    title: "Özel CNN",
    subtitle: "Alana özel · eşikli karar",
    accent: "from-neuro-violet/20 to-transparent",
  },
  {
    id: "cvat",
    title: "CVAT Cls+Seg",
    subtitle: "Yeni COCO · sınıf + maske",
    accent: "from-emerald-500/20 to-transparent",
  },
  {
    id: "compare",
    title: "İkili çalıştır",
    subtitle: "Yan yana karşılaştır",
    accent: "from-neuro-ice/15 to-neuro-violet/15",
  },
];

export function ModelSelector({
  value,
  onChange,
  disabled,
}: {
  value: RunMode;
  onChange: (m: RunMode) => void;
  disabled?: boolean;
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
      {options.map((opt) => {
        const on = value === opt.id;
        return (
          <motion.button
            key={opt.id}
            type="button"
            disabled={disabled}
            whileHover={disabled ? undefined : { y: -2 }}
            whileTap={disabled ? undefined : { scale: 0.98 }}
            onClick={() => onChange(opt.id)}
            className={`relative overflow-hidden rounded-xl border px-4 py-3 text-left transition-all ${
              on
                ? "border-neuro-cyan/40 bg-gradient-to-br shadow-neon ring-1 ring-neuro-cyan/30"
                : "border-white/[0.07] bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.04]"
            } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
          >
            <div
              className={`pointer-events-none absolute inset-0 bg-gradient-to-br opacity-60 ${opt.accent}`}
            />
            <div className="relative">
              <div className="font-display text-sm font-semibold text-white">
                {opt.title}
              </div>
              <div className="mt-0.5 text-[11px] text-slate-400 leading-snug">
                {opt.subtitle}
              </div>
            </div>
            {on && (
              <motion.div
                layoutId="model-pill"
                className="absolute bottom-2 right-2 h-1.5 w-1.5 rounded-full bg-neuro-cyan shadow-[0_0_8px_#2dd4bf]"
              />
            )}
          </motion.button>
        );
      })}
    </div>
  );
}
