import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import librosa

class AudioDataset(Dataset):
    def __init__(self, dataframe, target_sr=32000, clip_seconds=10, n_mels=128, image_size=224, transform=None):
        self.dataframe = dataframe.reset_index(drop=True) if isinstance(dataframe, pd.DataFrame) else pd.DataFrame(dataframe)
        if "path" not in self.dataframe.columns or "label" not in self.dataframe.columns:
            raise ValueError("AudioDataset expects a DataFrame with 'path' and 'label' columns.")

        self.target_sr = target_sr
        self.clip_seconds = clip_seconds
        self.n_mels = n_mels
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = str(row["path"])
        label = int(row["label"])

        audio_data, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)

        target_samples = int(self.target_sr * self.clip_seconds)
        if audio_data.shape[0] < target_samples:
            audio_data = np.pad(audio_data, (0, target_samples - audio_data.shape[0]))
        else:
            audio_data = audio_data[:target_samples]

        mel = librosa.feature.melspectrogram(y=audio_data, sr=self.target_sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.from_numpy(mel_db).float().unsqueeze(0)
        mel_tensor = (mel_tensor - mel_tensor.mean()) / (mel_tensor.std() + 1e-6)
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
