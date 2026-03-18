import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
import tensorflow as tf

from datasets import AudioDataset
from models import SimpleCNN, ResNet18, ResNet34
import preprocess as pp


def validate_master_list(df):
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


def train_test_split_master(df, val_ratio=0.2, test_ratio=0.2, group_col="track_id", random_state=21):
    total_holdout = val_ratio + test_ratio
    if not (0 < total_holdout < 1):
        raise ValueError("val_ratio + test_ratio must be between 0 and 1.")

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


def make_datasets(train_df, val_df, test_df):
    train_dataset = AudioDataset(train_df)
    val_dataset = AudioDataset(val_df)
    test_dataset = AudioDataset(test_df)
    return train_dataset, val_dataset, test_dataset


def make_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def build_model(model_name="resnet18", num_classes=2, pretrained=False):
    if model_name in {"simple_cnn", "simplecnn"}:
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == "resnet34":
        return ResNet34(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def build_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)


def build_scheduler(optimizer, step_size=10, gamma=0.1):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of the positive class
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")

    return epoch_loss, accuracy, f1, roc_auc


def compute_metrics(labels, preds, probs=None):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    scores = probs if probs is not None else preds
    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = float("nan")

    return accuracy, f1, roc_auc


def save_chkpt(model, optimizer, scheduler, epoch, path):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def fit(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=20, chkpt_path="checkpoint.pth"):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_roc_auc = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f} - Val ROC AUC: {val_roc_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_chkpt(model, optimizer, scheduler, epoch+1, chkpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        if scheduler is not None:
            scheduler.step()


def check_group_overlap(train_df, val_df, test_df, group_col="track_id"):
    if group_col not in train_df.columns:
        print("No group column for overlap check.")
        return

    train_groups = set(train_df[group_col].dropna().unique())
    val_groups = set(val_df[group_col].dropna().unique())
    test_groups = set(test_df[group_col].dropna().unique())

    print("train ∩ val:", len(train_groups & val_groups))
    print("train ∩ test:", len(train_groups & test_groups))
    print("val ∩ test:", len(val_groups & test_groups))


def main():
    # Collect and label all clips
    # 0 is fake, 1 is real
    master_list = pp.build_master_list(
        paths=["data/fake", "data/real"],
        labels=[0, 1],
        filetype=["wav", "mp3"]
    )

    validate_master_list(master_list)

    print(f"Total clips collected: {len(master_list)}")
    print(f"Label counts: {master_list['label'].value_counts().to_dict()}")

    if len(master_list) == 0:
        raise RuntimeError("No clips found. Check dataset paths and filetype.")

    train_df, val_df, test_df = train_test_split_master(master_list, val_ratio=0.2, test_ratio=0.2, group_col="track_id", random_state=21)

    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print("Train label counts:", train_df["label"].value_counts().to_dict())
    print("Val label counts:", val_df["label"].value_counts().to_dict())
    print("Test label counts:", test_df["label"].value_counts().to_dict())

    check_group_overlap(train_df, val_df, test_df, group_col="track_id")

    train_ds, val_ds, test_ds = make_datasets(train_df, val_df, test_df)
    train_loader, val_loader, test_loader = make_dataloaders(train_ds, val_ds, test_ds, batch_size=16, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("resnet18", num_classes=2, pretrained=False).to(device)

    log_dir = "logs/fit/" + pp.get_timestamp()
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    counts = train_df["label"].value_counts().to_dict()
    w0 = 1.0 / max(counts.get(0, 1), 1)
    w1 = 1.0 / max(counts.get(1, 1), 1)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = build_optimizer(model, learning_rate=1e-3)
    scheduler = build_scheduler(optimizer, step_size=5, gamma=0.5)

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=3,
        chkpt_path="checkpoint_debug.pth",
        callbacks=[tb_callback]
    )


if __name__ == "__main__":
    main()