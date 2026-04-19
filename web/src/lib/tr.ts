/** API İngilizce döndürse bile arayüzde gösterilecek Türkçe karşılıklar */
export function kararMetniEtiket(label: string): string {
  return label === "hemorrhage"
    ? "Kanama tespit edildi"
    : "Kanama tespit edilmedi";
}

export const GUVEN_ARALIGI_ACIKLAMA =
  "Yaklaşık %95 güven bandı (Wilson skoru, kanama olasılığı)";

/** API model anahtarlarını arayüzde göstermek için */
export function modelAnahtarTr(key: string): string {
  if (key === "pretrained") return "Önceden eğitilmiş";
  if (key === "custom") return "Özel CNN";
  if (key === "cvat") return "CVAT (Cls+Seg)";
  return key;
}
