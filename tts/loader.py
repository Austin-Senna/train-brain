import os
import glob
import pandas as pd    

def find_json(path):
    path = os.path.join(os.getcwd(), path)
    list_files = glob.glob(f'{path}/*.json?')
    list_files.sort()
    return list_files

def load(path, type):
    if type == "SYNTAX":
        return load_json_syntax(path)
    elif type == "SEMANTIC":
        return load_json_semantic(path)

def load_json_syntax(path):
    name = os.path.basename(path)
    df = pd.read_json(path_or_buf=path, lines=True)
    good = df['sentence_good'].tolist()
    bad = df['sentence_bad'].tolist()
    good =  [{'name': f"{name}_{i}_good", "audio": g} for i, g in enumerate(good)]
    bad =  [{'name': f"{name}_{i}_bad", "audio": b} for i, b in enumerate(bad)]
    return good + bad

def load_json_semantic(path):
    pass

def generate_voice(n, type):
    maps = []
    if type == "KOKORO":
        maps = ["af_heart", "af_bella", "am_michael", "am_fenrir"]
    elif type == "QWEN":
        maps = ["Ryan", "Aiden", "Sohee", "Ono_Anna"]
    results = []
    for i in range(n):
        results.append(maps[i%4])
    return results