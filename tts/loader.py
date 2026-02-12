import os
import glob
import pandas as pd    

def find_json(path):
    path = os.path.join(os.getcwd(), path)
    list_files = glob.glob(f'{path}/*.json?')
    list_files.sort()

    return list_files

def load_json(path):
    
    name = os.path.basename(path)
    df = pd.read_json(path_or_buf=path, lines=True)
    good = df['sentence_good'].tolist()
    bad = df['sentence_bad'].tolist()
    good =  [{'name': f"{name}_{i}_good", "audio": g} for i, g in enumerate(good)]
    bad =  [{'name': f"{name}_{i}_bad", "audio": b} for i, b in enumerate(bad)]
    return good + bad
   
def generate_voice(n):
    maps = ["Ryan", "Aiden", "Sohee", "Ono_Anna"]
    results = []
    for i in range(n):
        results.append(maps[i%4])
    return results