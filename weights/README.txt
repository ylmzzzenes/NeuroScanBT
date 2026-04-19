Eğitilmiş model ağırlıkları bu klasöre konur (dosya adları inference ile uyumlu):

  best_cls.pt   — scripts/train_classification.py çıktısından kopya
  best_unet.pt  — scripts/train_segmentation.py çıktısından kopya

Örnek (PowerShell, proje kökünden, eğitim runs/ klasörünüz varsa):

  copy runs\cls_coco\best_cls.pt weights\best_cls.pt
  copy runs\seg_coco\best_unet.pt weights\best_unet.pt

Alternatif: ortam değişkenleri NEURO_CLS_WEIGHTS ve NEURO_SEG_WEIGHTS ile özel yol.
Ağırlık dosyaları .gitignore ile repoya dahil edilmez; güvenli paylaşım için ayrı dağıtın.
