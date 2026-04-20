import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const ACCEPT = "image/png,image/jpeg,image/jpg,image/bmp,image/webp";

export function DropZone({
  file,
  previewUrl,
  onFile,
  busy,
}: {
  file: File | null;
  previewUrl: string | null;
  onFile: (f: File | null) => void;
  busy: boolean;
}) {
  const [drag, setDrag] = useState(false);
  const [uploadPulse, setUploadPulse] = useState(false);

  const handleFiles = useCallback(
    (list: FileList | null) => {
      if (!list?.length) return;
      const f = list[0];
      if (!f.type.startsWith("image/")) {
        onFile(null);
        return;
      }
      setUploadPulse(true);
      setTimeout(() => setUploadPulse(false), 600);
      onFile(f);
    },
    [onFile]
  );

  return (
    <div className="relative">
      <motion.label
        onDragEnter={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setDrag(false);
        }}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handleFiles(e.dataTransfer.files);
        }}
        animate={{
          borderColor: drag
            ? "rgba(45,212,191,0.45)"
            : "rgba(255,255,255,0.08)",
          boxShadow: drag
            ? "0 0 0 1px rgba(45,212,191,0.2), 0 12px 40px rgba(0,0,0,0.35)"
            : "0 4px 24px rgba(0,0,0,0.25)",
        }}
        className={`glass-panel flex min-h-[220px] cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed px-6 py-8 transition-colors ${
          busy ? "pointer-events-none opacity-60" : ""
        }`}
      >
        <input
          type="file"
          accept={ACCEPT}
          className="hidden"
          disabled={busy}
          onChange={(e) => handleFiles(e.target.files)}
        />
        <AnimatePresence mode="wait">
          {previewUrl ? (
            <motion.div
              key="prev"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0 }}
              className="relative w-full max-w-[280px]"
            >
              <div className="overflow-hidden rounded-lg ring-1 ring-white/10">
                <img
                  src={previewUrl}
                  alt="Önizleme"
                  className="mx-auto max-h-48 w-auto object-contain"
                />
              </div>
              <p className="mt-2 truncate text-center text-xs text-slate-500">
                {file?.name}
              </p>
            </motion.div>
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center"
            >
              <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-neuro-cyan/20 to-neuro-violet/10 ring-1 ring-white/10">
                <svg
                  className="h-7 w-7 text-neuro-ice"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
              </div>
              <p className="font-display text-sm font-medium text-slate-200">
                Beyin BT kesitini buraya bırakın
              </p>
              <p className="mt-1 text-xs text-slate-500">
                PNG · JPG · BMP · WebP
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.label>

      <motion.div
        className="pointer-events-none absolute inset-0 rounded-2xl"
        animate={{
          opacity: uploadPulse ? 0.35 : 0,
          scale: uploadPulse ? 1.02 : 1,
        }}
        style={{
          background:
            "radial-gradient(circle at 50% 50%, rgba(45,212,191,0.25), transparent 70%)",
        }}
      />

      {previewUrl && !busy && (
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            onFile(null);
          }}
          className="mt-3 w-full rounded-lg border border-white/10 bg-white/[0.03] py-2 text-xs text-slate-400 transition hover:border-rose-500/30 hover:text-rose-300"
        >
          Görüntüyü temizle
        </button>
      )}
    </div>
  );
}
