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
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
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


@app.on_event("startup")
def startup() -> None:
    _startup_load()
    if inference.mycnn_model is not None:
        print("MyCNN çıkarım ayarı:", inference.mycnn_inference_debug())


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
    debug_mycnn: bool = False,
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

    raw = predict_image_pil(pil, key, debug_raw_logits=debug_mycnn and api_model == "custom")
    heatmap_b64: str | None = None
    if include_heatmap:
        try:
            heatmap_b64 = grad_saliency_overlay_png(pil, key)
        except Exception:
            heatmap_b64 = None

    payload: dict[str, Any] = {
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
        "model_display_name": "ResNet18" if api_model == "pretrained" else "MyCNN",
    }
    if "mycnn_debug" in raw:
        payload["mycnn_debug"] = raw["mycnn_debug"]
    return payload


@app.get("/api/health")
def health() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ok": True,
        "device": str(inference.device),
        "models": {
            "pretrained": inference.resnet_model is not None,
            "custom": inference.mycnn_model is not None,
        },
    }
    if inference.mycnn_model is not None:
        out["mycnn"] = inference.mycnn_inference_debug()
    return out


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: Literal["pretrained", "custom"] = Form("pretrained"),
    include_heatmap: bool = Form(True),
    debug_mycnn: bool = Form(False),
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
        return _build_payload(rgb, model, include_heatmap, debug_mycnn=debug_mycnn)
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
