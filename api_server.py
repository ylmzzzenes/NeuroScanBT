"""
Beyin BT kanama tespiti REST API — FastAPI.
Çalıştırma: uvicorn api_server:app --reload --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import io
import traceback
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

import inference
import inference_coco
from inference import (
    ApiModelId,
    api_id_to_model_key,
    grad_saliency_overlay_png,
    predict_image_pil,
    risk_level_from_probs,
    wilson_ci,
)

app = FastAPI(title="NeuroScan AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _startup_load() -> None:
    try:
        inference.load_resnet()
    except Exception:
        inference.resnet_model = None
    try:
        inference.load_mycnn()
    except Exception:
        inference.mycnn_model = None
    try:
        inference_coco.load_coco_models()
    except Exception:
        inference_coco.cls_model = None
        inference_coco.seg_model = None


@app.on_event("startup")
def startup() -> None:
    _startup_load()


def _label_display(predicted_class: str) -> str:
    return (
        "Hemorrhage Detected"
        if predicted_class == "hemorrhage"
        else "No Hemorrhage Detected"
    )


def _build_payload(
    pil: Image.Image,
    api_model: ApiModelId,
    include_heatmap: bool,
) -> dict[str, Any]:
    key = api_id_to_model_key(api_model)
    if inference.model_for_key(key) is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": f"Model weights not loaded for {api_model}",
                "path": inference.RESNET_PATH if key == "ResNet18" else inference.MYCNN_PATH,
            },
        )

    raw = predict_image_pil(pil, key)
    heatmap_b64: str | None = None
    if include_heatmap:
        try:
            heatmap_b64 = grad_saliency_overlay_png(pil, key)
        except Exception:
            heatmap_b64 = None

    return {
        "model": api_model,
        "label": raw["predicted_class"],
        "label_display": _label_display(raw["predicted_class"]),
        "confidence_percent": raw["confidence_percent"],
        "hemorrhage_probability": raw["hemorrhage_probability"],
        "no_hemorrhage_probability": raw["no_hemorrhage_probability"],
        "confidence_interval": {
            "low": raw["confidence_interval_low"],
            "high": raw["confidence_interval_high"],
            "label": "95% approximate interval (Wilson score, hemorrhage probability)",
        },
        "risk_level": raw["risk_level"],
        "heatmap_png_base64": heatmap_b64,
        "segmentation_overlay_png_base64": None,
        "cls_seg_conflict": False,
        "model_display_name": "ResNet18" if api_model == "pretrained" else "MyCNN",
    }


def _build_payload_cvat(pil: Image.Image) -> dict[str, Any]:
    if inference_coco.cls_model is None or inference_coco.seg_model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": "CVAT (ResNet+U-Net) ağırlıkları yüklenemedi",
                "path_cls": inference_coco.CLS_PATH,
                "path_seg": inference_coco.SEG_PATH,
            },
        )
    raw = inference_coco.predict_cvat_pil(pil)
    lo, hi = wilson_ci(raw["hemorrhage_probability"])
    lo, hi = round(lo, 4), round(hi, 4)
    risk = risk_level_from_probs(
        raw["predicted_class"],
        raw["hemorrhage_probability"],
        raw["confidence_percent"],
    )
    return {
        "model": "cvat",
        "label": raw["predicted_class"],
        "label_display": _label_display(raw["predicted_class"]),
        "confidence_percent": raw["confidence_percent"],
        "hemorrhage_probability": raw["hemorrhage_probability"],
        "no_hemorrhage_probability": raw["no_hemorrhage_probability"],
        "confidence_interval": {
            "low": lo,
            "high": hi,
            "label": "95% approximate interval (Wilson score, hemorrhage probability)",
        },
        "risk_level": risk,
        "heatmap_png_base64": raw["segmentation_overlay_png_base64"],
        "segmentation_overlay_png_base64": raw["segmentation_overlay_png_base64"],
        "cls_seg_conflict": raw["cls_seg_conflict"],
        "model_display_name": raw["model_display_name"],
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "device": str(inference.device),
        "models": {
            "pretrained": inference.resnet_model is not None,
            "custom": inference.mycnn_model is not None,
            "cvat": inference_coco.cls_model is not None and inference_coco.seg_model is not None,
        },
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: Literal["pretrained", "custom", "cvat"] = Form("pretrained"),
    include_heatmap: bool = Form(True),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Geçerli bir görüntü dosyası yükleyin."},
        )
    try:
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={"code": "FILE_TOO_LARGE", "message": "Dosya çok büyük (max 25MB)."},
            )
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Dosya açılamadı veya geçersiz görüntü formatı."},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "UPLOAD_FAILED", "message": str(e)},
        )

    try:
        rgb = pil.convert("RGB")
        if model == "cvat":
            return _build_payload_cvat(rgb)
        return _build_payload(rgb, model, include_heatmap)
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"code": "INFERENCE_FAILED", "message": "Model çıkarımı başarısız."},
        )


@app.post("/api/predict/compare")
async def predict_compare(
    file: UploadFile = File(...),
    include_heatmap: bool = Form(True),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Geçerli bir görüntü dosyası yükleyin."},
        )
    try:
        data = await file.read()
        if len(data) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={"code": "FILE_TOO_LARGE", "message": "Dosya çok büyük (max 25MB)."},
            )
        pil = Image.open(io.BytesIO(data))
        pil.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_IMAGE", "message": "Dosya açılamadı veya geçersiz görüntü formatı."},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "UPLOAD_FAILED", "message": str(e)},
        )

    rgb = pil.convert("RGB")
    out: dict[str, Any] = {"pretrained": None, "custom": None, "errors": {}}
    for mid in ("pretrained", "custom"):
        try:
            out[mid] = _build_payload(rgb, mid, include_heatmap)  # type: ignore[arg-type]
        except HTTPException as e:
            out["errors"][mid] = e.detail
        except Exception:
            out["errors"][mid] = {"code": "INFERENCE_FAILED", "message": "Beklenmeyen hata"}
    return out
