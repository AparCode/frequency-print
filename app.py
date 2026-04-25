"""FastAPI server for browser-based audio deepfake inference."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from datasets import AudioDataset
from models import ResNet18, ResNet34, SimpleCNN


ROOT = Path(__file__).resolve().parent

CHECKPOINT_CANDIDATES = [
    ROOT / "checkpoints" / "current_ckpt" / "resnet34_20260421_231207" / "best_ckpt.pth",
    ROOT / "checkpoints" / "current_ckpt" / "resnet18_20260421_231154" / "best_ckpt.pth",
    ROOT / "checkpoints" / "current_ckpt" / "simplecnn_20260421_231117" / "best_ckpt.pth",
    ROOT / "checkpoints" / "resnet34" / "resnet34_20260413_193023" / "best_ckpt.pth",
    ROOT / "checkpoints" / "resnet18" / "resnet18_20260413_192905" / "best_ckpt.pth",
    ROOT / "checkpoints" / "simplecnn" / "simplecnn_20260413_192400" / "best_ckpt.pth",
]


def _resolve_checkpoint() -> Path:
    env_path = os.getenv("MODEL_CHECKPOINT")
    if env_path:
        env_ckpt = Path(env_path).expanduser().resolve()
        if not env_ckpt.exists():
            raise FileNotFoundError(f"MODEL_CHECKPOINT does not exist: {env_ckpt}")
        return env_ckpt

    for ckpt in CHECKPOINT_CANDIDATES:
        if ckpt.exists():
            return ckpt

    raise FileNotFoundError(
        "No checkpoint found. Set MODEL_CHECKPOINT to a valid .pth path."
    )


def _load_model_from_checkpoint(checkpoint_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt.get("model_name")
    if not model_name:
        raise ValueError(f"Checkpoint missing model_name: {checkpoint_path}")

    num_classes = int(ckpt.get("num_classes", 2))
    model = _build_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    threshold = float(ckpt.get("decision_threshold", 0.5))
    return model_name, model, threshold


def _build_model(model_name: str, num_classes: int = 2):
    model_key = str(model_name).lower()
    if model_key in {"simple_cnn", "simplecnn"}:
        return SimpleCNN(num_classes=num_classes)
    if model_key == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=False)
    if model_key == "resnet34":
        return ResNet34(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model_name in checkpoint: {model_name}")


class InferenceService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = _resolve_checkpoint()
        self.model_name, self.model, self.threshold = _load_model_from_checkpoint(
            self.checkpoint_path, self.device
        )

    def predict_file(self, audio_path: Path) -> dict:
        df = pd.DataFrame([{"path": str(audio_path), "label": 0}])
        dataset = AudioDataset(df, augment=False)
        inputs, _ = dataset[0]
        inputs = inputs.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)

        prob_real = float(probs[0, 1].item()) if probs.shape[1] > 1 else float(probs[0, 0].item())
        pred_idx = 1 if prob_real >= self.threshold else 0
        label = "real" if pred_idx == 1 else "fake"

        return {
            "label": label,
            "prediction": label,
            "confidence": prob_real,
            "pred_idx": pred_idx,
            "prob_real": prob_real,
            "threshold": self.threshold,
            "model": self.model_name,
            "checkpoint": str(self.checkpoint_path),
        }


app = FastAPI(title="FrequencyPrint API")
app.mount("/static", StaticFiles(directory=str(ROOT)), name="static")

service: InferenceService | None = None


@app.on_event("startup")
def _startup() -> None:
    global service
    service = InferenceService()


@app.get("/")
def read_index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health() -> JSONResponse:
    if service is None:
        return JSONResponse({"ok": False, "status": "starting"}, status_code=503)

    return JSONResponse(
        {
            "ok": True,
            "status": "ready",
            "device": service.device,
            "model": service.model_name,
            "checkpoint": str(service.checkpoint_path),
        }
    )


@app.post("/predict")
async def predict(audio: UploadFile = File(...)) -> JSONResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            temp_path = Path(tmp.name)
            content = await audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            tmp.write(content)

        result = service.predict_file(temp_path)
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    finally:
        await audio.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
