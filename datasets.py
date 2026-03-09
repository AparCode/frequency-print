from glob import glob
import torch
from torch.utils.data import Dataset
import librosa

audio_ldm = glob('data/fake_songs/FakeMusicCaps/audioldm2/*.wav')
# Print the first three paths of audio_ldm to verify
print(audio_ldm[:3])
music_gen = glob('data/fake_songs/FakeMusicCaps/MusicGen_medium/*.wav')
music_ldm = glob('data/fake_songs/FakeMusicCaps/musicldm/*.wav')
mus_tango = glob('data/fake_songs/FakeMusicCaps/mustango/*.wav')
stable_audio_open = glob('data/fake_songs/FakeMusicCaps/stable_audio_open/*.wav')

# Store the datasets in a dataloader
class AudioDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        print("DEBUG")
        # print(type(self.file_paths), len(self.file_paths))  # Debugging line to check the type of file_paths

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        audio_path = self.file_path[idx]
        # print(type(audio_path))  # Debugging line to check the type of audio_path
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        if self.transform:
            audio_data = self.transform(audio_data)
        return audio_data
