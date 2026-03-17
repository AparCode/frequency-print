import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models import SimpleCNN, ResNet18, ResNet34
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import argparse
from models import SimpleCNN, ResNet18, ResNet34
# from datasets import AudioDataset

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import preprocess as pp

if __name__ == "__main__":
    # Collect and label all clips
    # 0 is fake, 1 is real
    master_list = pp.build_master_list(
        paths=["data/fake", "data/real"],
        labels=[0, 1],
        filetype="wav"
    )

    # Process each clip and store the results in a new column
    waveforms = [pp.load_audio(path) for path in master_list["path"]]
    print(waveforms[0])