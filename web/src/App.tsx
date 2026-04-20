import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AnalyzingOverlay } from "./components/AnalyzingOverlay";
import { CompareGrid } from "./components/CompareGrid";
import { DropZone } from "./components/DropZone";
import { ModelSelector, type RunMode } from "./components/ModelSelector";
import { ResultCard } from "./components/ResultCard";
import { StepRail } from "./components/StepRail";
import type { CompareResponse, HealthResponse, PredictionPayload } from "./lib/api";
import { fetchHealth, predictCompare, predictImage } from "./lib/api";
import { modelAnahtarTr } from "./lib/tr";

const BACKEND_HINT =
  "API çalışmıyor. Proje klasöründe (web değil) yeni terminal: python -m uvicorn api_server:app --host 127.0.0.1 --port 8765";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthChecked, setHealthChecked] = useState(false);
  const [runMode, setRunMode] = useState<RunMode>("pretrained");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionPayload | null>(null);
  const [compare, setCompare] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setHealthChecked(true));
  }, []);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const activeStep = useMemo<1 | 2 | 3>(() => {
    if (result || (compare && (compare.pretrained || compare.custom))) return 3;
    if (file) return 2;
    return 1;
  }, [file, result, compare]);

  const onAnalyze = useCallback(async () => {
    setError(null);
    setResult(null);
    setCompare(null);
    if (!file) {
      setError("Önce geçerli bir görüntü yükleyin.");
      return;
    }

    let h = health;
    if (!h) {
      try {
        h = await fetchHealth();
        setHealth(h);
      } catch {
        setError(BACKEND_HINT);
        return;
      }
    }

    if (runMode === "pretrained" && !h.models.pretrained) {
      setError("Önceden eğitilmiş model sunucuda yüklü değil (.pth eksik).");
      return;
    }
    if (runMode === "custom" && !h.models.custom) {
      setError("Özel CNN sunucuda yüklü değil (.pth eksik).");
      return;
    }
    if (runMode === "compare" && !h.models.pretrained && !h.models.custom) {
      setError("Karşılaştırma için en az bir model gerekli.");
      return;
    }

    setLoading(true);
    try {
      if (runMode === "compare") {
        const c = await predictCompare(file, true);
        setCompare(c);
        if (!c.pretrained && !c.custom) {
          setError("Her iki model de çalıştırılamadı. Ağırlık dosyalarını kontrol edin.");
        }
      } else {
        const r = await predictImage(file, runMode, true);
        setResult(r);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analiz başarısız.");
    } finally {
      setLoading(false);
    }
  }, [file, runMode, health]);

  const modelReady = (m: RunMode) => {
    if (!health) return true;
    if (m === "compare") return health.models.pretrained || health.models.custom;
    if (m === "pretrained") return health.models.pretrained;
    return health.models.custom;
  };

  return (
    <div className="min-h-screen pb-20">
      <header className="border-b border-white/[0.06] bg-neuro-deep/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-neuro-cyan/80">
              Klinik yapay zekâ
            </p>
            <h1 className="font-display text-2xl font-bold text-white sm:text-3xl">
              Neuro<span className="text-gradient">Scan</span>
            </h1>
            <p className="mt-1 max-w-md text-sm text-slate-400">
              Beyin BT kanama taraması — iki model, kalibre belirsizlik ve dikkat
              haritası ile çıkarım.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {health ? (
              <>
                <StatusPill
                  ok={health.models.pretrained}
                  label="Önceden eğitilmiş"
                />
                <StatusPill ok={health.models.custom} label="Özel CNN" />
                <span className="rounded-lg border border-white/10 bg-white/[0.03] px-2 py-1 text-[10px] text-slate-500">
                  {health.device}
                </span>
              </>
            ) : (
              <span className="text-xs text-rose-300/90">
                API kapalı — sunucuyu başlatın
              </span>
            )}
          </div>
        </div>
      </header>

      {healthChecked && !health && (
        <div className="border-b border-amber-500/25 bg-amber-500/10 px-4 py-3 text-center text-sm text-amber-100">
          <strong className="font-semibold">Backend kapalı.</strong>{" "}
          <span className="text-amber-100/85">{BACKEND_HINT}</span>
        </div>
      )}

      <main className="mx-auto max-w-6xl px-4 pt-8">
        <div className="glass-panel p-4 sm:p-6">
          <StepRail activeStep={activeStep} />
        </div>

        <div className="mt-8 grid gap-8 lg:grid-cols-12 lg:gap-10">
          <section className="lg:col-span-5 space-y-6">
            <div>
              <h2 className="font-display text-lg font-semibold text-white">
                1 · Çıkarım yolunu seçin
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Model değişince eşikler, etiketler ve ısı haritası buna göre
                güncellenir.
              </p>
              <div className="mt-4">
                <ModelSelector
                  value={runMode}
                  onChange={(m) => {
                    setRunMode(m);
                    setResult(null);
                    setCompare(null);
                    setError(null);
                  }}
                  disabled={loading}
                />
              </div>
              {!modelReady(runMode) && (
                <p className="mt-3 text-xs text-amber-200/90">
                  Bu seçenek için model dosyası eksik — sunucu klasörüne{" "}
                  <code className="rounded bg-white/10 px-1">.pth</code> ekleyin.
                </p>
              )}
            </div>

            <div>
              <h2 className="font-display text-lg font-semibold text-white">
                2 · Görüntüyü yükleyin
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Kesiti sürükleyip bırakın veya dosya seçin — görüntüler diske
                kaydedilmez.
              </p>
              <div className="mt-4">
                <DropZone
                  file={file}
                  previewUrl={previewUrl}
                  onFile={(f) => {
                    setFile(f);
                    setResult(null);
                    setCompare(null);
                    setError(null);
                  }}
                  busy={loading}
                />
              </div>
            </div>

            <motion.button
              type="button"
              disabled={loading || !file}
              whileHover={{ scale: loading || !file ? 1 : 1.01 }}
              whileTap={{ scale: loading || !file ? 1 : 0.99 }}
              onClick={onAnalyze}
              className="w-full rounded-xl bg-gradient-to-r from-neuro-cyan to-teal-500 py-3.5 font-display text-sm font-semibold text-neuro-void shadow-neon transition disabled:opacity-40 disabled:shadow-none"
            >
              {runMode === "compare"
                ? "İki modeli birlikte çalıştır"
                : "Yapay zekâ analizini başlat"}
            </motion.button>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          <section className="lg:col-span-7 space-y-4">
            <h2 className="font-display text-lg font-semibold text-white">
              3 · Klinik özet
            </h2>
            <p className="text-sm text-slate-500">
              Karar, güven bandı, risk düzeyi ve dikkat (saliency) katmanı.
            </p>

            {!result && !compare && !loading && (
              <div className="glass-panel flex min-h-[320px] flex-col items-center justify-center gap-2 p-8 text-center">
                <div className="h-12 w-12 rounded-full border border-dashed border-white/15" />
                <p className="text-sm text-slate-500">
                  Modeli seçin, görüntüyü yükleyin — sonuç burada görünecek.
                </p>
              </div>
            )}

            {compare && (
              <>
                <CompareGrid data={compare} />
                {Object.keys(compare.errors).length > 0 && (
                  <div className="rounded-xl border border-amber-500/25 bg-amber-500/5 p-4 text-xs text-amber-100/90">
                    <p className="font-semibold">Kısmi çalıştırma</p>
                    <ul className="mt-2 list-inside list-disc space-y-1 text-amber-100/70">
                      {Object.entries(compare.errors).map(([k, v]) => (
                        <li key={k}>
                          {modelAnahtarTr(k)}: {JSON.stringify(v)}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            )}

            {result && !compare && (
              <ResultCard data={result} title="Seçilen model" />
            )}
          </section>
        </div>
      </main>

      <footer className="mx-auto mt-16 max-w-6xl px-4 pb-8 text-center text-[11px] text-slate-600">
        Yardımcı yapay zekâdır — radyolog değerlendirmesinin yerini tutmaz.
      </footer>

      <AnimatePresence>
        {loading && <AnalyzingOverlay key="analyzing" />}
      </AnimatePresence>
    </div>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium ${
        ok
          ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
          : "border-white/10 bg-white/[0.03] text-slate-500"
      }`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-emerald-400 shadow-[0_0_6px_#34d399]" : "bg-slate-600"}`}
      />
      {label}
    </span>
  );
}
