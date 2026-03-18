"""Preprocessing utilities for manifest creation and basic audio loading."""

from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio


MANIFEST_COLUMNS = ["path", "label", "source_dataset", "generator_family", "track_id"]
IGNORED_PATH_PARTS = {"__MACOSX"}


def _infer_source_metadata(file_path: Path):
    """Infer source and generator metadata from a file path."""
    parts = file_path.parts
    parts_lower = [p.lower() for p in parts]

    if "fakemusiccaps" in parts_lower:
        idx = parts_lower.index("fakemusiccaps")
        generator = parts[idx + 1] if idx + 1 < len(parts) - 1 else "unknown"
        return "FakeMusicCaps", generator

    if "real" in parts_lower:
        return "real", "real"

    if "fake" in parts_lower:
        return "fake", "unknown"

    return "unknown", "unknown"


def _normalize_filetypes(filetype, num_paths):
    """Normalize a filetype argument to a list matching the number of paths."""
    if isinstance(filetype, (list, tuple)):
        if len(filetype) != num_paths:
            raise ValueError("When filetype is a list/tuple, it must match len(paths).")
        return [str(ext).lstrip(".") for ext in filetype]

    return [str(filetype).lstrip(".")] * num_paths

# Collect and label all clips
def collect_clips(path, label, filetype="wav"):
    clips = []

    pattern = f"*.{str(filetype).lstrip('.')}"
    for file in Path(path).rglob(pattern):
        if any(part in IGNORED_PATH_PARTS for part in file.parts):
            continue

        file_path = str(file)
        source_dataset, generator_family = _infer_source_metadata(file)
        track_id = file.stem  # filename without extension

        clips.append((file_path, label, source_dataset, generator_family, track_id))

    return pd.DataFrame(clips, columns=MANIFEST_COLUMNS)


# Building one master list of all clips and labels
def build_master_list(paths, labels, filetype="wav"):
    if len(paths) != len(labels):
        raise ValueError("paths and labels must have the same length.")

    filetypes = _normalize_filetypes(filetype, len(paths))

    master_frames = []
    for path, label, ext in zip(paths, labels, filetypes):
        df = collect_clips(path, label, ext)
        master_frames.append(df)

    if not master_frames:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)

    return pd.concat(master_frames, ignore_index=True)

# Load audio and convert it to a waveform array
def load_audio(file_path):
    try:
        audio, sr = torchaudio.load(file_path)
    except Exception:
        # Fallback path when torchaudio backend dependencies are unavailable.
        wav, sr = sf.read(file_path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(wav.T)

    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    return audio, sr

# Resample to one sample rate
def resample_audio(audio, orig_sr, target_sr=32000):
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        audio = resampler(audio)
    return audio, target_sr

# Remove only leading and trailing silence
def trim_edge_silence(audio, sr, top_db=20):
    pass

# Make every clip 10 seconds long
def trim_clip_length(audio, sr, target_length=10):
    pass

# Normalize loudness to -23 LUFS
def normalize_loudness(audio, sr, target_lufs=-23): 
    pass

# Catch bad clips and return None
def validate_clip(audio, sr):
    pass

# Process one clip through all steps
def process_clip(file_path, target_sr=32000, target_length=10, target_lufs=-23):
    pass

def assign_splits():
    pass

def save_master_list():
    pass
