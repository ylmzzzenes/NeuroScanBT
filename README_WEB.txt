NeuroScan web arayüzü (Vite + React + Tailwind)
==============================================

`web/` alt projesi FastAPI backend’e bağlanır. Geliştirmede Vite, `/api`
isteklerini `http://127.0.0.1:8765` adresine proxy’ler (web/vite.config.ts).

Kurulum — Python API (proje kökü: enes/)
  pip install -r requirements.txt
  Ağırlık dosyaları (isteğe bağlı env ile yol verilebilir):
    NEUROSCAN_RESNET_PATH → ResNet18 ikili sınıflandırıcı (.pth)
    NEUROSCAN_MYCNN_PATH → MyCNN (.pth), yoksa ./best_mycnn_v2.pth denenir
  Varsayılan ResNet dosya adı: ./resnet18_brain.pth

Kurulum — web (bir kez)
  cd web
  npm install

Çalıştırma (aynı anda iki terminal)
  Terminal 1 — proje kökünde:
    python -m uvicorn api_server:app --host 127.0.0.1 --port 8765
  Terminal 2:
    cd web
    npm run dev
  Tarayıcı: http://localhost:5173

  `vite preview` (üretim önizleme, port 4173) için de aynı `/api` proxy
  tanımlıdır; API yine 8765’te çalışmalıdır.

API adresi farklıysa `web/.env` oluşturun:
  VITE_API_BASE=http://127.0.0.1:8766

Üretim derlemesi
  cd web
  npm run build
  Çıktı: web/dist/

Not: `arayuz.py` Tkinter masaüstü arayüzüdür; `web/` tarayıcı arayüzüdür.
