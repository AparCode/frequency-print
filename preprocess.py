# -----------------
# PREPROCESSING
# -----------------

import pathlib
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torchaudio
import pyloudnorm as pyln
import tqdm

# Collect and label all clips
def collect_clips(path, label, filetype="wav"):
    clips = []

    # create a table with these columns
    # path, label, source_dataset, generator_family, track_id
    for file in pathlib.Path(path).rglob(f"*.{filetype}"):
        if "__MACOSX" in file.parts:
            continue

        file_path = str(file)
        source_dataset = "unknown"
        generator_family = "unknown"

        if "FakeMusicCaps" in file.parts:
            source_dataset = "FakeMusicCaps"
            idx = file.parts.index("FakeMusicCaps")
            if idx + 1 < len(file.parts) - 1:
                generator_family = file.parts[idx + 1]
        elif "real" in file.parts:
            source_dataset = "real"
            generator_family = "real"
        elif "fake" in file.parts:
            source_dataset = "fake"

        track_id = file.stem  # filename without extension

        clips.append((file_path, label, source_dataset, generator_family, track_id))
    return pd.DataFrame(clips, columns=["path", "label", "source_dataset", "generator_family", "track_id"])


# Building one master master list of all clips and labels
def build_master_list(paths, labels, filetype="wav"):
    if isinstance(filetype, (list, tuple)):
        if len(filetype) != len(paths):
            raise ValueError("When filetype is a list/tuple, it must match len(paths).")
        filetypes = list(filetype)
    else:
        filetypes = [filetype] * len(paths)

    master_list = []
    for path, label, ext in zip(paths, labels, filetypes):
        df = collect_clips(path, label, ext)
        master_list.append(df)

    if len(master_list) == 0:
        return pd.DataFrame(columns=["path", "label", "source_dataset", "generator_family", "track_id"])

    return pd.concat(master_list, ignore_index=True)

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
    # try:
    #     audio, sr = load_audio(file_path)
    #     audio, sr = resample_audio(audio, sr, target_sr)
    #     audio = trim_edge_silence(audio, sr)
    #     audio = trim_clip_length(audio, sr, target_length)
    #     audio = normalize_loudness(audio, sr, target_lufs)
    #     if validate_clip(audio, sr):
    #         return audio
    #     else:
    #         return None
    # except Exception as e:
    #     print(f"Error processing {file_path}: {e}")
    #     return None 

def assign_splits():
    pass

def save_master_list():
    pass
