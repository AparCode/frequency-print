import os
import argparse
import numpy as np
from typing import Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from tqdm.auto import tqdm

from datasets import AudioDataset
from models import SimpleCNN, ResNet18, ResNet34
import preprocess as pp
import scan_bad_audio

# Checks to see if the Master List is valid through many checks:
# 1. Has the required columns
# 2. Is not empty
# 3. Has both classes represented
# 4. All file paths exist

def validate_master_list(df) -> None:
    """Validate manifest integrity before split/training."""
    required_columns = {"path", "label", "source_dataset", "generator_family", "track_id"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Master list is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Master list is empty.")

    if df["label"].nunique() < 2:
        raise ValueError("Master list needs both classes. Check file extensions/paths for real and fake data.")

    missing_paths = (~df["path"].map(os.path.exists)).sum()
    if missing_paths > 0:
        raise ValueError(f"Master list has {missing_paths} missing file paths.")

    print("Master list validation passed.")


def _split_generator_holdout(
    df,
    val_ratio=0.2,
    test_ratio=0.2,
    random_state=21,
    holdout_generator: Optional[str] = None,
):
    """Split by holding out one fake generator family for the test split.

    This produces a tougher domain-shift evaluation while still keeping both
    classes in train/val/test by splitting real clips independently.
    """
    fake_df = df[df["label"] == 0].copy()
    real_df = df[df["label"] == 1].copy()

    if fake_df.empty or real_df.empty:
        raise ValueError("generator_holdout split requires both fake and real samples.")

    if "generator_family" not in fake_df.columns:
        raise ValueError("generator_holdout split requires 'generator_family' column.")

    gen_counts = fake_df["generator_family"].value_counts()
    if gen_counts.shape[0] < 2:
        raise ValueError(
            "generator_holdout split needs at least two fake generator families "
            "so one can be held out and one remains for training."
        )

    if holdout_generator is None:
        # Default to the largest generator family for a strong out-of-domain test.
        holdout_generator = str(gen_counts.idxmax())

    if holdout_generator not in set(fake_df["generator_family"].astype(str).unique()):
        raise ValueError(
            f"Holdout generator '{holdout_generator}' not found in fake data. "
            f"Available: {sorted(fake_df['generator_family'].astype(str).unique().tolist())}"
        )

    fake_holdout = fake_df[fake_df["generator_family"].astype(str) == holdout_generator].copy()
    fake_remaining = fake_df[fake_df["generator_family"].astype(str) != holdout_generator].copy()

    if fake_remaining.empty:
        raise ValueError("No fake samples remain for train/val after holdout selection.")

    # Keep val proportion relative to the non-test portion of the data.
    val_fraction_of_non_test = val_ratio / (1.0 - test_ratio)

    fake_train, fake_val = train_test_split(
        fake_remaining,
        test_size=val_fraction_of_non_test,
        random_state=random_state,
        shuffle=True,
    )

    real_train_val, real_test = train_test_split(
        real_df,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,
    )
    real_train, real_val = train_test_split(
        real_train_val,
        test_size=val_fraction_of_non_test,
        random_state=random_state,
        shuffle=True,
    )

    train_df = (
        np.random.RandomState(random_state)
        .permutation(pd_concat([fake_train, real_train]).index)
    )
    val_df = (
        np.random.RandomState(random_state + 1)
        .permutation(pd_concat([fake_val, real_val]).index)
    )
    test_df = (
        np.random.RandomState(random_state + 2)
        .permutation(pd_concat([fake_holdout, real_test]).index)
    )

    train_combined = pd_concat([fake_train, real_train]).loc[train_df].reset_index(drop=True)
    val_combined = pd_concat([fake_val, real_val]).loc[val_df].reset_index(drop=True)
    test_combined = pd_concat([fake_holdout, real_test]).loc[test_df].reset_index(drop=True)

    for split_name, split_df in [("train", train_combined), ("val", val_combined), ("test", test_combined)]:
        if split_df["label"].nunique() < 2:
            raise ValueError(f"{split_name} split is single-class under generator_holdout; adjust settings.")

    print(f"generator_holdout split active. held-out fake generator: {holdout_generator}")
    return train_combined, val_combined, test_combined


def pd_concat(frames):
    """Small helper so concat intent reads clearly where used repeatedly."""
    import pandas as pd

    return pd.concat(frames, axis=0)


# Helper function for splitting the master list
def train_test_split_master(
    df,
    val_ratio=0.2,
    test_ratio=0.2,
    group_col="track_id",
    random_state=21,
    split_mode="group",
    holdout_generator: Optional[str] = None,
):
    """Split manifest into train/val/test using selected split strategy.

    split_mode='group' uses group-aware splitting by group_col when possible.
    split_mode='generator_holdout' holds out one fake generator family for test.
    """
    total_holdout = val_ratio + test_ratio
    if not (0 < total_holdout < 1):
        raise ValueError("val_ratio + test_ratio must be between 0 and 1.")

    if split_mode == "generator_holdout":
        return _split_generator_holdout(
            df,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            holdout_generator=holdout_generator,
        )

    if split_mode != "group":
        raise ValueError("split_mode must be one of: 'group', 'generator_holdout'.")

    use_groups = (
        group_col in df.columns
        and df[group_col].notna().all()
        and df[group_col].nunique() > 1
    )

    if use_groups:
        gss = GroupShuffleSplit(n_splits=1, test_size=total_holdout, random_state=random_state)
        train_idx, temp_idx = next(gss.split(df, y=df["label"], groups=df[group_col]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        temp_df = df.iloc[temp_idx].reset_index(drop=True)

        test_fraction_of_temp = test_ratio / total_holdout
        can_group_split_temp = (
            group_col in temp_df.columns
            and temp_df[group_col].notna().all()
            and temp_df[group_col].nunique() > 1
        )

        if can_group_split_temp:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=test_fraction_of_temp, random_state=random_state)
            val_idx, test_idx = next(gss2.split(temp_df, y=temp_df["label"], groups=temp_df[group_col]))
            val_df = temp_df.iloc[val_idx].reset_index(drop=True)
            test_df = temp_df.iloc[test_idx].reset_index(drop=True)
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_fraction_of_temp,
                random_state=random_state,
                stratify=temp_df["label"] if temp_df["label"].nunique() > 1 else None,
            )
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=total_holdout,
            random_state=random_state,
            stratify=df["label"] if df["label"].nunique() > 1 else None,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / total_holdout,
            random_state=random_state,
            stratify=temp_df["label"] if temp_df["label"].nunique() > 1 else None,
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Create datasets using the dataframes.
def make_datasets(train_df, val_df, test_df):
    """Construct dataset objects for each split."""
    train_dataset = AudioDataset(train_df, augment=True)
    val_dataset = AudioDataset(val_df, augment=False)
    test_dataset = AudioDataset(test_df, augment=False)
    return train_dataset, val_dataset, test_dataset

# Create dataloaders from the datasets.
def make_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0, pin_memory=False):
    """Create split dataloaders with optional worker/prefetch settings."""
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

# Select the best checkpoint by validation and learn the decision threshold on the validation set.
def find_best_threshold(labels, probs, metric="f1", min_t=0.05, max_t=0.95, n_steps=91):
    best_t = 0.5
    best_score = -1.0
    thresholds = np.linspace(min_t, max_t, n_steps)
    
    for t in thresholds:
        preds = (np.array(probs) >= t).astype(int)
        if metric == "f1":
            score = f1_score(labels, preds, zero_division=0)
        elif metric == "balanced_accuracy":
            score = balanced_accuracy_score(labels, preds)
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_t = float(t)
        
    return best_t, float(best_score)

# Build the model based on the specified architecture and parameters.
def build_model(model_name="resnet18", num_classes=2, pretrained=False):
    """Return a model instance for the given architecture name."""
    if model_name in {"simple_cnn", "simplecnn"}:
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Build the optimizer
def build_optimizer(model, learning_rate=1e-3):
    """Build the optimizer used for training."""
    return optim.Adam(model.parameters(), lr=learning_rate)

# Build the scheduler
def build_scheduler(optimizer, step_size=10, gamma=0.1):
    """Build the learning-rate scheduler used during fit."""
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Train one epoch at a time
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_amp=False):
    """Run one training epoch and return average loss."""
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Validate one epoch at a time, returning loss and metrics
def validate_one_epoch(model, dataloader, criterion, device, use_amp=False):
    """Run one validation epoch and return loss + core metrics."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of the positive class
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    val_bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        val_roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_roc_auc = float("nan")

    best_t, best_f1 = find_best_threshold(all_labels, all_probs, metric="f1")
    return epoch_loss, val_acc, val_f1, val_bal_acc, val_roc_auc, best_t, best_f1

# Compute metrics given labels, predictions, and optionally probabilities
def compute_metrics(labels, preds, probs=None):
    """Compute accuracy, F1, and ROC AUC from labels/predictions."""
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    scores = probs if probs is not None else preds
    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = float("nan")

    return accuracy, f1, roc_auc


# Records a checkpoint of the model, optimizer, scheduler, and epoch number to the specified path.
def save_chkpt(model, optimizer, scheduler, epoch, path):
    """Save resumable training checkpoint state."""
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)

def save_infer_chkpt(model, epoch, best_val_loss, model_name, num_classes=2, path="checkpoint.pth", best_f1=None, decision_threshold=0.5):
    """Save lightweight inference checkpoint state."""
    state = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": num_classes,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_f1": best_f1,
        "decision_threshold": decision_threshold
    }
    torch.save(state, path)

# Train the model
def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=20, best_ckpt_path="best_checkpoint.pth", last_ckpt_path="last_checkpoint.pth", log_dir=None, model_name="simplecnn", num_classes=2):
    """Train a model across epochs while logging and checkpointing."""
    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None
    best_val_loss = float("inf")
    best_f1 = -1.0
    best_threshold = 0.5
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, use_amp=use_amp)
        val_loss, val_acc, val_f1, val_bal_acc, val_roc_auc, epoch_best_t, epoch_best_f1 = validate_one_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
        if epoch_best_f1 > best_f1:
            best_f1 = epoch_best_f1
            best_threshold = epoch_best_t
            save_infer_chkpt(
                model=model,
                epoch=epoch + 1,
                best_val_loss=best_val_loss,
                model_name=model_name,
                num_classes=num_classes,
                path=best_ckpt_path,
                best_f1=best_f1,
                decision_threshold=best_threshold
            )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("loss/val", val_loss, epoch + 1)
            writer.add_scalar("metrics/val_acc", val_acc, epoch + 1)
            writer.add_scalar("metrics/val_f1", val_f1, epoch + 1)
            writer.add_scalar("metrics/val_roc_auc", val_roc_auc, epoch + 1)

        if scheduler is not None:
            scheduler.step()

    if writer is not None:
        writer.close()

# Check for group overlap between train, val, and test sets based on the specified group column.
def check_group_overlap(train_df, val_df, test_df, group_col="track_id"):
    """Compute group-overlap sets across splits (debug helper)."""
    if group_col not in train_df.columns:
        print("No group column for overlap check.")
        return

    train_groups = set(train_df[group_col].dropna().unique())
    val_groups = set(val_df[group_col].dropna().unique())
    test_groups = set(test_df[group_col].dropna().unique())

    # print("train ∩ val:", len(train_groups & val_groups))
    # print("train ∩ test:", len(train_groups & test_groups))
    # print("val ∩ test:", len(val_groups & test_groups))


def preflight_audio_scan(master_list, scan_limit=0):
    """Scan audio decode health before training and report risky files."""
    scan_paths = master_list["path"].tolist()
    if scan_limit and scan_limit > 0:
        scan_paths = scan_paths[:scan_limit]

    bad_rows = []
    fallback_rows = []
    for audio_path in scan_paths:
        status, message = scan_bad_audio.check_file(audio_path)
        if status == "bad":
            bad_rows.append((audio_path, message))
        elif status == "fallback":
            fallback_rows.append((audio_path, message))

    if bad_rows:
        print(f"Audio preflight scan found {len(bad_rows)} unreadable files.")
        for audio_path, message in bad_rows[:10]:
            print(f"  BAD: {audio_path}")
            print(f"       {message}")
    else:
        print("Audio preflight scan found no unreadable files.")

    if fallback_rows:
        print(f"Audio preflight scan found {len(fallback_rows)} files that require ffmpeg fallback.")

    return bad_rows, fallback_rows

# The Main Function
def main(
    model_type: str,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    scan_audio: bool = True,
    scan_limit: int = 0,
    split_mode: str = "group",
    holdout_generator: Optional[str] = None,
) -> None:
    """Entry point for a full training run."""
    # Collect and label all clips
    # 0 is fake, 1 is real
    master_list = pp.build_master_list(
        paths=["data/fake", "data/real"],
        labels=[0, 1],
        filetype=["wav", "mp3"]
    )

    validate_master_list(master_list)

    # print(f"Total clips collected: {len(master_list)}")
    # print(f"Label counts: {master_list['label'].value_counts().to_dict()}")

    if len(master_list) == 0:
        raise RuntimeError("No clips found. Check dataset paths and filetype.")

    if scan_audio:
        preflight_audio_scan(master_list, scan_limit=scan_limit)

    # Use the chosen split strategy. generator_holdout is useful for domain-shift testing.
    train_df, val_df, test_df = train_test_split_master(
        master_list,
        val_ratio=0.2,
        test_ratio=0.2,
        group_col="track_id",
        random_state=35,
        split_mode=split_mode,
        holdout_generator=holdout_generator,
    )

    # print("Split sizes:", len(train_df), len(val_df), len(test_df))
    # print("Train label counts:", train_df["label"].value_counts().to_dict())
    # print("Val label counts:", val_df["label"].value_counts().to_dict())
    # print("Test label counts:", test_df["label"].value_counts().to_dict())

    check_group_overlap(train_df, val_df, test_df, group_col="track_id")

    train_ds, val_ds, test_ds = make_datasets(train_df, val_df, test_df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    if pin_memory is None:
        pin_memory = (device == "cuda")

    train_loader, val_loader, _ = make_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # print(model_type)
    model = build_model(model_type, num_classes=2, pretrained=False).to(device)

    counts = train_df["label"].value_counts().to_dict()
    w0 = 1.0 / max(counts.get(0, 1), 1)
    w1 = 1.0 / max(counts.get(1, 1), 1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, learning_rate=1e-3)
    scheduler = build_scheduler(optimizer, step_size=5, gamma=0.5)
    
    run_id = pp.get_timestamp()
    run_name = f"{model_type}_{run_id}"
    log_dir = os.path.join("logs", "fit", run_name)

    ckpt_dir = os.path.join("checkpoints", model_type, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pth")
    last_ckpt_path = os.path.join(ckpt_dir, "last_ckpt.pth")

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=20,
        best_ckpt_path=best_ckpt_path,
        last_ckpt_path=last_ckpt_path,
        log_dir=log_dir,
        model_name=model_type,
        num_classes=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model to classify real vs fake audio clips.")
    parser.add_argument('-m', '--model', type=str, default="simplecnn", help="Model architecture: simple_cnn, resnet18, or resnet34")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--num_workers', type=int, default=max(1, (os.cpu_count() or 2) // 2), help="DataLoader worker processes")
    parser.add_argument('--pin_memory', action='store_true', help="Enable pinned-memory DataLoader transfers")
    parser.add_argument('--no_pin_memory', action='store_true', help="Disable pinned-memory DataLoader transfers")
    parser.add_argument('--no_audio_scan', action='store_true', help="Skip the audio preflight scan")
    parser.add_argument('--audio_scan_limit', type=int, default=0, help="Limit preflight scan to the first N files; 0 scans all files")
    parser.add_argument('--split_mode', type=str, choices=['group', 'generator_holdout'], default='group', help="Split strategy: group-aware split or fake-generator holdout")
    parser.add_argument('--holdout_generator', type=str, default=None, help="Fake generator_family to hold out when --split_mode generator_holdout")
    args = parser.parse_args()

    if args.pin_memory and args.no_pin_memory:
        raise ValueError("Use only one of --pin_memory or --no_pin_memory")

    if args.pin_memory:
        pin_memory = True
    elif args.no_pin_memory:
        pin_memory = False
    else:
        pin_memory = None

    model = args.model
    main(
        model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        scan_audio=not args.no_audio_scan,
        scan_limit=args.audio_scan_limit,
        split_mode=args.split_mode,
        holdout_generator=args.holdout_generator,
    )