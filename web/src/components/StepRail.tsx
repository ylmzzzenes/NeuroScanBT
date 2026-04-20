import { motion } from "framer-motion";

const steps = [
  { id: 1, title: "Model", hint: "Mimari seçimi" },
  { id: 2, title: "Yükleme", hint: "Beyin BT" },
  { id: 3, title: "Sonuç", hint: "YZ kararı" },
];

export function StepRail({ activeStep }: { activeStep: 1 | 2 | 3 }) {
  return (
    <div className="flex items-center justify-between gap-2 sm:gap-4">
      {steps.map((s, i) => {
        const done = activeStep > s.id;
        const current = activeStep === s.id;
        return (
          <div key={s.id} className="flex flex-1 items-center gap-2 min-w-0">
            <div className="flex flex-col items-center flex-1 min-w-0">
              <motion.div
                initial={false}
                animate={{
                  scale: current ? 1.05 : 1,
                  boxShadow: current
                    ? "0 0 20px rgba(45,212,191,0.35)"
                    : "0 0 0 rgba(0,0,0,0)",
                }}
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border text-xs font-semibold font-display transition-colors ${
                  done
                    ? "border-neuro-cyan/50 bg-neuro-cyan/15 text-neuro-ice"
                    : current
                      ? "border-neuro-cyan bg-neuro-cyan/20 text-white"
                      : "border-white/10 bg-white/[0.03] text-slate-500"
                }`}
              >
                {done ? "✓" : s.id}
              </motion.div>
              <span
                className={`mt-2 text-[10px] sm:text-xs font-medium uppercase tracking-wider truncate max-w-full ${
                  current ? "text-neuro-ice" : "text-slate-500"
                }`}
              >
                {s.title}
              </span>
              <span className="hidden sm:block text-[10px] text-slate-600 truncate">
                {s.hint}
              </span>
            </div>
            {i < steps.length - 1 && (
              <div
                className={`h-px flex-1 mb-6 min-w-[12px] rounded-full ${
                  activeStep > s.id
                    ? "bg-gradient-to-r from-neuro-cyan/60 to-neuro-violet/40"
                    : "bg-white/[0.06]"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
