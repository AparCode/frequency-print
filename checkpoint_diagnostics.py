import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import preprocess as pp
from train import AudioDataset, DataLoader, build_model, train_test_split_master


DEFAULT_CHECKPOINTS = [
    "checkpoints/simplecnn/simplecnn_20260407_183623/best_ckpt.pth",
    "checkpoints/resnet18/resnet18_20260407_183724/best_ckpt.pth",
    "checkpoints/resnet34/resnet34_20260407_183741/best_ckpt.pth",
]


def infer_fallback_model_name(ckpt_path: str) -> str:
    """Infer model architecture from checkpoint path segments."""
    p = Path(ckpt_path)
    parts = [x.lower() for x in p.parts]
    for candidate in ["simplecnn", "resnet18", "resnet34"]:
        if candidate in parts:
            return candidate
    return "resnet18"


def label_counts(df) -> Counter:
    """Return label frequency counts for a split DataFrame."""
    return Counter(df["label"].tolist())


def split_and_report(
    master_df,
    val_ratio,
    test_ratio,
    group_col,
    random_state,
    split_mode="group",
    holdout_generator: Optional[str] = None,
):
    """Split data and print per-split class counts.

    split_mode mirrors train.py so diagnostics can validate the same split policy.
    """
    train_df, val_df, test_df = train_test_split_master(
        master_df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_col=group_col,
        random_state=random_state,
        split_mode=split_mode,
        holdout_generator=holdout_generator,
    )

    train_counts = label_counts(train_df)
    val_counts = label_counts(val_df)
    test_counts = label_counts(test_df)

    print("Split sizes and label counts")
    print(f"train: {len(train_df)} {dict(train_counts)}")
    print(f"val:   {len(val_df)} {dict(val_counts)}")
    print(f"test:  {len(test_df)} {dict(test_counts)}")
    print("")

    return train_df, val_df, test_df


def passes_minimum_per_class(counts: Counter, minimum_per_class: int) -> bool:
    """Check whether both classes satisfy minimum-count threshold."""
    return counts.get(0, 0) >= minimum_per_class and counts.get(1, 0) >= minimum_per_class


def find_seed_with_minimums(
    master_df,
    val_ratio,
    test_ratio,
    group_col,
    minimum_per_class,
    max_tries=500,
    split_mode="group",
    holdout_generator: Optional[str] = None,
):
    """Search for a split seed that satisfies minimum class counts."""
    for seed in range(max_tries):
        _, _, test_df = train_test_split_master(
            master_df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            group_col=group_col,
            random_state=seed,
            split_mode=split_mode,
            holdout_generator=holdout_generator,
        )
        counts = label_counts(test_df)
        if passes_minimum_per_class(counts, minimum_per_class):
            return seed, counts
    return None, None


def evaluate_checkpoint(ckpt_path, test_loader, device):
    """Evaluate one checkpoint and print robust metrics."""
    ckpt = torch.load(ckpt_path, map_location=device)
    fallback_name = infer_fallback_model_name(ckpt_path)
    model_name = ckpt.get("model_name", fallback_name)
    num_classes = int(ckpt.get("num_classes", 2))

    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, zero_division=0, labels=[0, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    try:
        roc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) == 2 else float("nan")
    except ValueError:
        roc = float("nan")

    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    majority_baseline = max(true_counts.values()) / len(y_true)

    print("=" * 72)
    print(f"model: {model_name}")
    print(f"checkpoint: {ckpt_path}")
    print(f"accuracy: {acc:.6f}")
    print(f"balanced_accuracy: {bal_acc:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    print(f"roc_auc: {roc:.6f}" if not np.isnan(roc) else "roc_auc: nan")
    print(f"true_counts: {dict(true_counts)}")
    print(f"pred_counts: {dict(pred_counts)}")
    print(f"majority_baseline_accuracy: {majority_baseline:.6f}")
    print("confusion_matrix [[tn, fp], [fn, tp]]:")
    print(cm)
    print(f"per_class_precision [class0, class1]: {per_class[0].tolist()}")
    print(f"per_class_recall [class0, class1]: {per_class[1].tolist()}")
    print(f"per_class_f1 [class0, class1]: {per_class[2].tolist()}")

    if len(true_counts) < 2:
        print("WARNING: Only one class present in y_true, so accuracy is misleading.")
    elif acc <= majority_baseline + 1e-12:
        print("WARNING: Accuracy is at or below the majority-class baseline.")


def parse_args():
    """Parse command-line arguments for diagnostics workflow."""
    parser = argparse.ArgumentParser(
        description="Diagnose split imbalance and evaluate checkpoints with robust metrics."
    )
    parser.add_argument("--fake_dir", default="data/fake", help="Path to fake audio root")
    parser.add_argument("--real_dir", default="data/real", help="Path to real audio root")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--group_col", default="track_id")
    parser.add_argument(
        "--split_mode",
        choices=["group", "generator_holdout"],
        default="group",
        help="Split strategy used in diagnostics (mirrors train.py).",
    )
    parser.add_argument(
        "--holdout_generator",
        default=None,
        help="generator_family to hold out when --split_mode generator_holdout",
    )
    parser.add_argument("--seed", type=int, default=35, help="Split random_state")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_per_class", type=int, default=50, help="Minimum required examples of each class in test split")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="One or more checkpoint paths",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if test split fails minimum-per-class check",
    )
    parser.add_argument(
        "--find_seed",
        action="store_true",
        help="Search for a split seed that satisfies minimum-per-class test counts",
    )
    parser.add_argument(
        "--max_seed_tries",
        type=int,
        default=500,
        help="How many seeds to try when --find_seed is set",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only run split diagnostics and optional seed search",
    )
    return parser.parse_args()


def main() -> None:
    """Run split diagnostics and optional checkpoint evaluation."""
    args = parse_args()

    master_df = pp.build_master_list(
        paths=[args.fake_dir, args.real_dir],
        labels=[0, 1],
        filetype=["wav", "mp3"],
    )

    _, _, test_df = split_and_report(
        master_df,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        group_col=args.group_col,
        random_state=args.seed,
        split_mode=args.split_mode,
        holdout_generator=args.holdout_generator,
    )

    test_counts = label_counts(test_df)
    split_ok = passes_minimum_per_class(test_counts, args.min_per_class)

    if not split_ok:
        print(
            "WARNING: Test split fails minimum-per-class check "
            f"(need at least {args.min_per_class} of each class, got {dict(test_counts)})."
        )
        if args.find_seed:
            # Seed search is useful for group mode; for generator_holdout it only
            # changes the real-class split because fake holdout is deterministic.
            found_seed, found_counts = find_seed_with_minimums(
                master_df,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                group_col=args.group_col,
                minimum_per_class=args.min_per_class,
                max_tries=args.max_seed_tries,
                split_mode=args.split_mode,
                holdout_generator=args.holdout_generator,
            )
            if found_seed is None:
                print(f"No valid seed found within 0..{args.max_seed_tries - 1}.")
            else:
                print(
                    "Suggested seed that satisfies minimum test counts: "
                    f"{found_seed} with counts {dict(found_counts)}"
                )

        if args.strict:
            raise SystemExit(2)
    else:
        print("Test split passes minimum-per-class check.")

    if args.skip_eval:
        return

    test_loader = DataLoader(
        AudioDataset(test_df),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt_path in args.checkpoints:
        evaluate_checkpoint(ckpt_path, test_loader=test_loader, device=device)


if __name__ == "__main__":
    main()
