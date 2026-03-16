# -----------------
# PREPROCESSING
# -----------------

import pathlib
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio
import pyloudnorm as pyln
import tqdm

# Collect and label all clips
def collect_clips(path, label, filetype="wav"):
    clips = []
    for file in pathlib.Path(path).rglob(f"*.{filetype}"):
        clips.append((str(file), label))
    return clips


