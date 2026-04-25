"""Run checkpoint inference on a folder of YouTube audio snippets."""

import argparse
import csv
from html import parser
import math
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import AudioDataset
from train import build_model


DEFAULT_CHECKPOINTS = [
    "checkpoints/simplecnn/simplecnn_20260413_192400/best_ckpt.pth",
    "checkpoints/resnet18/resnet18_20260413_192905/best_ckpt.pth",
    "checkpoints/resnet34/resnet34_20260413_193023/best_ckpt.pth",
]


def resolve_inputs(input_dir: str, exts: list[str]) -> list[str]:
    """Resolve and deduplicate input audio files by extension."""
    base = Path(input_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}. "
            "Create it first or pass --input_dir with the correct path."
        )

    files = []
    for ext in exts:
        files.extend(base.rglob(f"*.{ext.lower()}"))
        files.extend(base.rglob(f"*.{ext.upper()}"))

    deduped = sorted({str(p) for p in files})
    if not deduped:
        raise RuntimeError(f"No audio files found in {input_dir} for extensions {exts}")
    return deduped


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    override_model: Optional[str] = None,
):
    """Load a checkpoint and return a ready-to-run model."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = override_model if override_model else ckpt.get("model_name")
    if not model_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no model_name. "
            "Pass --model_override with simplecnn, resnet18, or resnet34."
        )

    num_classes = int(ckpt.get("num_classes", 2))
    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    threshold = float(ckpt.get("decision_threshold", 0.5))
    return model_name, model, threshold

def _tensor_to_image(spec_tensor: torch.Tensor) -> np.ndarray:
    # spec_tensor shape is [3, H, W]; channels are repeated in dataset preprocessing.
    single = spec_tensor[0].detach().cpu().numpy()
    lo, hi = float(single.min()), float(single.max())
    if hi - lo < 1e-12:
        return np.zeros_like(single, dtype=np.float32)
    return ((single - lo) / (hi - lo)).astype(np.float32)


def _save_spectrogram(spec_tensor: torch.Tensor, save_path: Path, title: str) -> None:
    img = _tensor_to_image(spec_tensor)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Time Elapsed (in seconds)")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def _confidence_bucket(prob_real: float, threshold: float, margin: float) -> str:
    # distance from decision boundary
    dist = abs(prob_real - threshold)
    return "high_conf" if dist >= margin else "low_conf"


def run_inference(
    files: list[str],
    checkpoints: list[str],
    output_csv: str,
    batch_size: int = 64,
    num_workers: int = 4,
    model_override: Optional[str] = None,
    save_specs: bool = False,
    specs_dir: str = "yt_vids/specs",
    max_specs_per_model: int = 300,
    conf_margin: float = 0.20,
) -> None:
    """Run batched inference for all checkpoints and write CSV output."""
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dummy label is required by AudioDataset interface.
    df = pd.DataFrame([{"path": path, "label": 0} for path in files])
    dataset = AudioDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    loaded = []
    t_model_load_start = time.perf_counter()
    for ckpt_path in checkpoints:
        model_name, model, threshold = load_model_from_checkpoint(ckpt_path, device=device, override_model=model_override)
        loaded.append((ckpt_path, model_name, model, threshold))
    t_model_load = time.perf_counter() - t_model_load_start

    total_batches = math.ceil(len(files) / batch_size)
    print(f"Device: {device}")
    print(f"Files: {len(files)} | Models: {len(loaded)} | Batch size: {batch_size} | Batches: {total_batches}")
    print(f"Model load time: {t_model_load:.2f}s")

    rows = []
    index_base = 0
    t_infer_start = time.perf_counter()
    
    saved_per_model = {}
    specs_root = Path(specs_dir)
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Batches", total=total_batches):
            inputs = inputs.to(device, non_blocking=True)
            batch_n = inputs.size(0)
            batch_paths = files[index_base : index_base + batch_n]

            for ckpt_path, model_name, model, threshold in loaded:
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).cpu()
                probs_pos = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                preds = (probs_pos >= threshold).long().cpu().tolist()
                probs_real = probs_pos.tolist()
                inputs_cpu = inputs.detach().cpu()

                model_key = f"{model_name}_{Path(ckpt_path).parent.name}"
                if model_key not in saved_per_model:
                    saved_per_model[model_key] = 0

                for i, path in enumerate(batch_paths):
                    conf_bucket = _confidence_bucket(float(probs_real[i]), float(threshold), float(conf_margin))
                    pred_label = "real" if preds[i] == 1 else "fake"

                    spec_path_str = ""
                    if save_specs and saved_per_model[model_key] < max_specs_per_model:
                        file_stem = Path(path).stem
                        spec_name = f"{file_stem}_p{pred_label}_r{probs_real[i]:.3f}.png"
                        spec_path = specs_root / model_key / f"predicted_{pred_label}" / conf_bucket / spec_name
                        title = f"Spectrogram Analysis for {file_stem} using {model_name}"
                        _save_spectrogram(inputs_cpu[i], spec_path, title)
                        spec_path_str = str(spec_path)
                        saved_per_model[model_key] += 1
                        
                    p0 = float(probs[i, 0].item()) if probs.shape[1] > 0 else float("nan")
                    p1 = float(probs[i, 1].item()) if probs.shape[1] > 1 else float("nan")
                    rows.append(
                        {
                            "file": path,
                            "checkpoint": ckpt_path,
                            "model": model_name,
                            "pred_idx": preds[i],
                            "pred_label": "fake" if preds[i] == 0 else "real",
                            "prob_fake": p0,
                            "prob_real": p1,
                            "conf_bucket": conf_bucket,
                            "spec_path": spec_path_str,
                        }
                    )

            index_base += batch_n
    t_infer = time.perf_counter() - t_infer_start

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "checkpoint",
                "model",
                "pred_idx",
                "pred_label",
                "prob_fake",
                "prob_real",
                "conf_bucket",
                "spec_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    t_total = time.perf_counter() - t0

    print(f"Wrote {len(rows)} prediction rows to {out_path}")
    print(f"Input files: {len(files)}")
    print(f"Models: {len(loaded)}")
    print(f"Inference time: {t_infer:.2f}s")
    print(f"Total runtime: {t_total:.2f}s")


def main() -> None:
    """CLI entry point for YouTube snippet inference."""
    parser = argparse.ArgumentParser(description="Run inference on YouTube audio snippets and export predictions to CSV.")
    parser.add_argument("--input_dir", type=str, default="youtube_snippets", help="Directory containing snippet audio files")
    parser.add_argument("--exts", nargs="+", default=["mp3", "wav", "m4a"], help="Audio extensions to include")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS, help="Checkpoint paths to evaluate")
    parser.add_argument("--output_csv", type=str, default="youtube_snippets/predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--model_override", type=str, default=None, help="Force model type if checkpoint lacks model_name")
    parser.add_argument("--save_specs", action="store_true", help="Save model-input spectrogram images during inference")
    parser.add_argument("--specs_dir", type=str, default="yt_vids/specs", help="Directory for spectrogram exports")
    parser.add_argument("--max_specs_per_model", type=int, default=300, help="Max spectrogram images per model")
    parser.add_argument("--conf_margin", type=float, default=0.20, help="Distance from threshold for high_conf bucket")
    args = parser.parse_args()

    files = resolve_inputs(args.input_dir, args.exts)
    run_inference(
        files=files,
        checkpoints=args.checkpoints,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_override=args.model_override,
        save_specs=args.save_specs,
        specs_dir=args.specs_dir,
        max_specs_per_model=args.max_specs_per_model,
        conf_margin=args.conf_margin
    )


if __name__ == "__main__":
    main()
