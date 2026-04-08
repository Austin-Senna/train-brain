"""TTS comps_converted/comps_wugs_dist.jsonl with Kokoro and emit metadata."""
import os
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from kokoro import KPipeline

INPUT_JSONL = "/projects/bgbh/awijaya/train-brain/tts/comps_converted/comps_wugs_dist.jsonl"
OUTPUT_DIR = "/projects/bgbh/targeted-embedding/datasets/COMPS_KOKORO/comps_wugs_dist"
METADATA_OUT = "/projects/bgbh/targeted-embedding/metadata/comps_wugs_dist_metadata_master.json"
SPLIT_NAME = "comps_wugs_dist"
VOICES = ["af_heart", "af_bella", "am_michael", "am_fenrir"]
WORKERS = 4

worker_pipeline = None

def init_worker():
    global worker_pipeline
    worker_pipeline = KPipeline(lang_code='a')

def process_audio_task(task):
    text, voice, save_path = task
    if os.path.exists(save_path):
        return
    try:
        chunks = []
        for _, _, audio in worker_pipeline(text, voice=voice):
            chunks.append(audio)
        if chunks:
            sf.write(save_path, np.concatenate(chunks), 24000)
    except Exception as e:
        print(f"Error generating {save_path}: {e}")

def build_tasks_and_metadata():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tasks = []
    metadata = []
    with open(INPUT_JSONL) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for cond_key, cond_label in [("sentence_good", "good"), ("sentence_bad", "bad")]:
                fname = f"comps_wugs_dist_{i}_{cond_label}.wav"
                save_path = os.path.join(OUTPUT_DIR, fname)
                voice = VOICES[(i * 2 + (0 if cond_label == "good" else 1)) % len(VOICES)]
                text = row[cond_key]
                tasks.append((text, voice, save_path))
                metadata.append({
                    "filename": fname,
                    "filepath": save_path,
                    "split": SPLIT_NAME,
                    "pair_index": i,
                    "condition": cond_label,
                    "sentence_text": text,
                    "contrast_sentence": row["sentence_bad" if cond_key == "sentence_good" else "sentence_good"],
                })
    return tasks, metadata

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    tasks, metadata = build_tasks_and_metadata()
    print(f"Total tasks: {len(tasks)}")
    pending = [t for t in tasks if not os.path.exists(t[2])]
    print(f"Pending generation: {len(pending)}")

    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
            list(ex.map(process_audio_task, pending))

    os.makedirs(os.path.dirname(METADATA_OUT), exist_ok=True)
    with open(METADATA_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {METADATA_OUT} ({len(metadata)} entries)")
    print("Done.")
