import os
import glob
from io import BytesIO
import librosa

def load(path):
    path = os.path.join("/projects/bgbh/datasets", path)
    list_files = glob.glob(f'{path}/*.wav')
    list_files.sort()

    if not list_files:
        print(f"Warning: No files found in {path}")

    return list_files

def audio_generator(file_list, processor = None):
    target_sr = processor.feature_extractor.sampling_rate if processor else None
    for file_name in file_list:
        audio, _ = librosa.load(file_name, sr=target_sr)
        yield audio, file_name  

def find_json(path):
    path = os.path.join(os.getcwd(), path)
    list_files = glob.glob(f'{path}/*.json?')
    list_files.sort()

    return list_files

def load_json(path):
    import pandas as pd    
    name = os.path.basename(path)
    df = pd.read_json(path_or_buf=path, lines=True)
    
    sentences = df[['sentence_good', 'sentence_bad']].values.tolist()
    return [{"id": f"{name}_{i}", "good": sentence[0], "bad": sentence[1]} for i, sentence in enumerate(sentences)]
