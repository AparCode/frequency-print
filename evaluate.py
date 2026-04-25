"""Evaluate trained checkpoints and inspect checkpoint internals."""
import csv
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

import torch

import preprocess as pp
from train import AudioDataset, DataLoader, build_model, train_test_split_master


def infer_model_name_from_checkpoint(ckpt: dict[str, Any], cli_model_name: Optional[str] = None) -> str:
    """Resolve the model architecture name from CLI override or checkpoint metadata."""
    if cli_model_name:
        return cli_model_name
    if "model_name" in ckpt:
        return ckpt["model_name"]
    raise ValueError(
        "Could not determine model architecture from checkpoint. "
        "Pass --model (simplecnn, resnet18, or resnet34)."
    )


def inspect_checkpoint(ckpt: dict[str, Any], param_name: Optional[str] = None, limit: int = 30) -> None:
    """Print checkpoint metadata and optionally a specific parameter tensor summary."""
    print("Checkpoint keys:", list(ckpt.keys()))
    for key in ["model_name", "num_classes", "epoch", "best_val_loss"]:
        if key in ckpt:
            print(f"{key}: {ckpt[key]}")

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        print("No model_state_dict found in checkpoint.")
        return

    names = list(state_dict.keys())
    print(f"\nTotal parameter tensors: {len(names)}")

    if param_name:
        if param_name not in state_dict:
            print(f"Parameter '{param_name}' not found.")
            return
        tensor = state_dict[param_name]
        print(f"\nParameter: {param_name}")
        print(f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        print(
            f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
            f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}"
        )
        flat = tensor.flatten()
        n_show = min(10, flat.numel())
        print(f"first_{n_show}_values={flat[:n_show].tolist()}")
        return

    print("\nFirst parameter keys:")
    for name in names[:limit]:
        print(name)

    weight_names = [n for n in names if n.endswith("weight")]
    bias_names = [n for n in names if n.endswith("bias")]
    print(f"\nFound {len(weight_names)} weight tensors and {len(bias_names)} bias tensors.")


def resolve_pin_memory_arg(pin_memory: bool, no_pin_memory: bool) -> Optional[bool]:
    """Resolve mutually exclusive pin-memory flags into an optional boolean override."""
    if pin_memory and no_pin_memory:
        raise ValueError("Use only one of --pin_memory or --no_pin_memory")
    if pin_memory:
        return True
    if no_pin_memory:
        return False
    return None

def outcome_bucket(true_label: int, pred_label: int) -> str:
    """Categorize a prediction outcome into TP, TN, FP, or FN."""
    if true_label == 1 and pred_label == 1:
        return "TP"
    elif true_label == 0 and pred_label == 0:
        return "TN"
    elif true_label == 0 and pred_label == 1:
        return "FP"
    elif true_label == 1 and pred_label == 0:
        return "FN"
    else:
        raise ValueError(f"Invalid label combination: true={true_label}, pred={pred_label}")

def tensor_to_plot_image(spec_tensor: torch.Tensor) -> np.ndarray:
    # spec_tensor is [3, H, W] from your dataset; channels are repeated copies.
    single = spec_tensor[0].detach().cpu().numpy()

    # Input was z-score normalized in dataset; map to 0-1 for display.
    lo, hi = float(single.min()), float(single.max())
    if hi - lo < 1e-12:
        return np.zeros_like(single, dtype=np.float32)
    return ((single - lo) / (hi - lo)).astype(np.float32)


def save_spec_png(spec_tensor: torch.Tensor, save_path: Path, title: str) -> None:
    img = tensor_to_plot_image(spec_tensor)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
def evaluate(
    model_path: str,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    save_specs: bool = False,
    specs_dir: str = "eval_artifacts",
    max_specs_per_bucket: int = 40,
) -> None:
    """Evaluate a checkpoint on the project test split and print class-aware diagnostics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'.")

    resolved_model_name = infer_model_name_from_checkpoint(ckpt, cli_model_name=model_name)
    resolved_num_classes = int(num_classes if num_classes is not None else ckpt.get("num_classes", 2))

    model = build_model(resolved_model_name, num_classes=resolved_num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Recreate the test split to evaluate the model
    master_list = pp.build_master_list(paths=["data/fake", "data/real"], labels=[0, 1], filetype=["wav", "mp3"])
    _, _, test_list = train_test_split_master(
        master_list,
        val_ratio=0.2,
        test_ratio=0.2,
        group_col="track_id",
        random_state=35,
    )
    test_dataset = AudioDataset(test_list)
    if pin_memory is None:
        pin_memory = (device == "cuda")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    threshold = float(ckpt.get("decision_threshold", 0.5))
    
    rows = []
    saved_per_bucket = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    artifact_root = Path(specs_dir)
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            probs_pos = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs_pos >= threshold).long()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            labels_cpu = labels.detach().cpu()
            preds_cpu = preds.detach().cpu()
            probs_cpu = probs_pos.detach().cpu()
            inputs_cpu = inputs.detach().cpu()

            for i in range(labels_cpu.shape[0]):
                true_i = int(labels_cpu[i].item())
                pred_i = int(preds_cpu[i].item())
                conf_i = float(probs_cpu[i].item())
                bucket = outcome_bucket(true_i, pred_i)

                sample_id = f"b{batch_idx:04d}_i{i:02d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "true_label": true_i,
                        "pred_label": pred_i,
                        "confidence_pos": conf_i,
                        "bucket": bucket,
                    }
                )

                if save_specs and saved_per_bucket[bucket] < max_specs_per_bucket:
                    save_name = f"{sample_id}_t{true_i}_p{pred_i}_c{conf_i:.3f}.png"
                    save_path = artifact_root / bucket / save_name
                    title = f"{bucket} | true={true_i} pred={pred_i} conf={conf_i:.3f}"
                    save_spec_png(inputs_cpu[i], save_path, title)
                    saved_per_bucket[bucket] += 1

    accuracy = correct / total if total > 0 else 0.0
    true_counts = Counter(all_labels)
    pred_counts = Counter(all_preds)
    majority_baseline = (max(true_counts.values()) / total) if total > 0 and true_counts else 0.0

    print(f"Model: {resolved_model_name}, num_classes: {resolved_num_classes}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"True label counts: {dict(true_counts)}")
    print(f"Pred label counts: {dict(pred_counts)}")
    print(f"Majority-class baseline accuracy: {majority_baseline:.4f}")

    if len(true_counts) < 2:
        print("WARNING: Test set contains only one class. Accuracy is not reliable for model quality.")
    elif accuracy <= majority_baseline + 1e-9:
        print("WARNING: Accuracy is at or below majority-class baseline.")
    
    if rows:
        artifact_root.mkdir(parents=True, exist_ok=True)
        csv_path = artifact_root / "predictions.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sample_id", "true_label", "pred_label", "confidence_pos", "bucket"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions table to: {csv_path}")

    if save_specs:
        print(f"Saved spectrograms under: {artifact_root}")
        print(f"Per-bucket counts: {saved_per_bucket}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model or inspect a checkpoint.")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Model architecture: simplecnn, resnet18, or resnet34")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of output classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--pin_memory", action="store_true", help="Enable pinned memory in DataLoader")
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable pinned memory in DataLoader")
    parser.add_argument("--inspect", action="store_true", help="Print checkpoint/training metadata and parameter keys")
    parser.add_argument("--param", type=str, default=None, help="Specific parameter key to inspect (e.g., model.conv1.weight)")
    parser.add_argument("--limit", type=int, default=30, help="How many parameter keys to print in inspect mode")
    parser.add_argument("--save_specs", action="store_true", help="Save spectrogram images by TP/TN/FP/FN")
    parser.add_argument("--specs_dir", type=str, default="eval_artifacts", help="Directory for eval images and CSV")
    parser.add_argument("--max_specs_per_bucket", type=int, default=40, help="Max images to save per TP/TN/FP/FN")
    args = parser.parse_args()

    if args.inspect or args.param is not None:
        ckpt_for_inspection = torch.load(args.model_path, map_location="cpu")
        inspect_checkpoint(ckpt_for_inspection, param_name=args.param, limit=args.limit)
    else:
        pin_memory = resolve_pin_memory_arg(args.pin_memory, args.no_pin_memory)

        evaluate(
            args.model_path,
            model_name=args.model,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            save_specs=args.save_specs,
            specs_dir=args.specs_dir,
            max_specs_per_bucket=args.max_specs_per_bucket
        )
