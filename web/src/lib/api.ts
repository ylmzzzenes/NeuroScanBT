const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type ModelId = "pretrained" | "custom";

export interface PredictionPayload {
  model: ModelId;
  label: string;
  label_display: string;
  confidence_percent: number;
  hemorrhage_probability: number;
  no_hemorrhage_probability: number;
  confidence_interval: {
    low: number;
    high: number;
    label: string;
  };
  risk_level: "low" | "medium" | "high";
  heatmap_png_base64: string | null;
  /** CVAT yolu: kanama maskesi üzerine bindirme (base64 PNG) */
  segmentation_overlay_png_base64?: string | null;
  /** Sınıflandırma kanama derken maskenin neredeyse boş olduğu tutarsızlık bayrağı */
  cls_seg_conflict?: boolean;
  /** İnsan okunaklı model adı */
  model_display_name?: string;
  /** CVAT: segmentasyon PNG gerçekten üretildi mi */
  segmentation_overlay_shown?: boolean;
  /** Geliştirici: sınıf olasılıkları, indeksler, karar kuralı */
  debug?: Record<string, unknown>;
}

export interface HealthResponse {
  ok: boolean;
  device: string;
  models: { pretrained: boolean; custom: boolean };
}

export interface CompareResponse {
  pretrained: PredictionPayload | null;
  custom: PredictionPayload | null;
  errors: Partial<Record<ModelId, unknown>>;
}

function detailMessage(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) {
    const m = (detail as { message?: string }).message;
    if (typeof m === "string") return m;
  }
  return "İstek başarısız oldu.";
}

function mapNetworkError(e: unknown): Error {
  if (e instanceof TypeError || (e instanceof Error && e.name === "TypeError")) {
    return new Error(
      "Sunucuya bağlanılamıyor. Proje kökünde API'yi başlatın: python -m uvicorn api_server:app --host 127.0.0.1 --port 8765"
    );
  }
  return e instanceof Error ? e : new Error("İstek başarısız oldu.");
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) throw new Error("API yanıt vermiyor.");
    return r.json();
  } catch (e) {
    throw mapNetworkError(e);
  }
}

export async function predictImage(
  file: File,
  model: ModelId,
  includeHeatmap = true
): Promise<PredictionPayload> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("model", model);
  fd.append("include_heatmap", String(includeHeatmap));

  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: fd });
  } catch (e) {
    throw mapNetworkError(e);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(detailMessage(data.detail ?? data));
  }
  return data as PredictionPayload;
}

export async function predictCompare(
  file: File,
  includeHeatmap = true
): Promise<CompareResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("include_heatmap", String(includeHeatmap));

  let r: Response;
  try {
    r = await fetch(`${API_BASE}/api/predict/compare`, {
      method: "POST",
      body: fd,
    });
  } catch (e) {
    throw mapNetworkError(e);
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    throw new Error(detailMessage(data.detail ?? data));
  }
  return data as CompareResponse;
}
