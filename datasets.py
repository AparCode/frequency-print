import os
import shutil
import subprocess
from contextlib import contextmanager
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", message=".*Xing stream size off by more than 1%.*")

_FAILED_DECODE_PATHS = set()


@contextmanager
def _suppress_stderr_fd():
    # Some audio backends print directly to stderr file descriptor 2.
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _load_audio_ffmpeg(audio_path, target_sr):
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg is not available in PATH")

    cmd = [
        ffmpeg_bin,
        "-v",
        "error",
        "-nostdin",
        "-i",
        audio_path,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        stderr_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(stderr_msg or "ffmpeg failed to decode audio")

    audio_data = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio_data.size == 0:
        raise RuntimeError("Decoded empty audio stream")
    return audio_data, target_sr


def _load_audio_robust(audio_path, target_sr):
    try:
        with _suppress_stderr_fd():
            return librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception:
        return _load_audio_ffmpeg(audio_path, target_sr)


class AudioDataset(Dataset):
    def __init__(self, dataframe, target_sr=32000, clip_seconds=10, n_mels=128, image_size=224, transform=None, augment=False):
        self.dataframe = dataframe.reset_index(drop=True) if isinstance(dataframe, pd.DataFrame) else pd.DataFrame(dataframe)
        if "path" not in self.dataframe.columns or "label" not in self.dataframe.columns:
            raise ValueError("AudioDataset expects a DataFrame with 'path' and 'label' columns.")

        self.target_sr = target_sr
        self.clip_seconds = clip_seconds
        self.n_mels = n_mels
        self.image_size = image_size
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataframe)

    def _spec_augment(self, mel_tensor):
        # mel_tensor has shape [1, n_mels, time]
        if torch.rand(1).item() < 0.8:
            f = torch.randint(0, 16, (1,)).item()
            if f > 0:
                f0 = torch.randint(0, max(1, mel_tensor.shape[1] - f + 1), (1,)).item()
                mel_tensor[:, f0:f0 + f, :] = 0.0

        if torch.rand(1).item() < 0.8:
            t = torch.randint(0, 32, (1,)).item()
            if t > 0:
                t0 = torch.randint(0, max(1, mel_tensor.shape[2] - t + 1), (1,)).item()
                mel_tensor[:, :, t0:t0 + t] = 0.0

        return mel_tensor

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = str(row["path"])
        label = int(row["label"])

        try:
            audio_data, _ = _load_audio_robust(audio_path, self.target_sr)
        except Exception as decode_err:
            if audio_path not in _FAILED_DECODE_PATHS:
                warnings.warn(f"Failed to decode {audio_path}: {decode_err}. Using silence for this sample.")
                _FAILED_DECODE_PATHS.add(audio_path)
            audio_data = np.zeros(int(self.target_sr * self.clip_seconds), dtype=np.float32)

        target_samples = int(self.target_sr * self.clip_seconds)
        if audio_data.shape[0] < target_samples:
            audio_data = np.pad(audio_data, (0, target_samples - audio_data.shape[0]))
        else:
            if self.augment:
                max_start = audio_data.shape[0] - target_samples
                start = int(torch.randint(0, max_start + 1, (1,)).item()) if max_start > 0 else 0
            else:
                start = max(0, (audio_data.shape[0] - target_samples) // 2)
            audio_data = audio_data[start:start + target_samples]

        mel = librosa.feature.melspectrogram(y=audio_data, sr=self.target_sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.from_numpy(mel_db).float().unsqueeze(0)
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)

        if self.augment:
            mel_tensor = self._spec_augment(mel_tensor)

        mel_tensor = F.interpolate(
            mel_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mel_tensor = mel_tensor.repeat(3, 1, 1)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, torch.tensor(label, dtype=torch.long)
