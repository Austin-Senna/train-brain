import os
import glob
from io import BytesIO
import librosa

def load(path):
    cwd = os.getcwd()
    path = os.path.join(cwd, path)
    list_files = glob.glob(f'{path}/*')

    if not list_files:
        print(f"Warning: No files found in {path}")

    return list_files

def audio_generator(file_list, processor = None):
    target_sr = processor.feature_extractor.sampling_rate if processor else None
    for file_name in file_list:
        audio, _ = librosa.load(file_name, sr=target_sr)
        yield audio, file_name  