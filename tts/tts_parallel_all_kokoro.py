# parallel is after irregular plural susbject verb agreeement
import os
import torch
import soundfile as sf
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from loader import find_json, load, generate_voice
from kokoro import KPipeline

TYPE = "SYNTAX"
output_prefix = "BLIMP_KOKORO"

# Global variable for worker processes so the model isn't reloaded constantly
worker_pipeline = None

def init_worker():
    """Initializes the TTS pipeline once per worker process."""
    global worker_pipeline
    worker_pipeline = KPipeline(lang_code='a')

def process_audio_task(task):
    """Worker function to generate and save a single audio file."""
    text, speaker, save_path = task
    
    # Skip if file already exists
    if os.path.exists(save_path):
        return

    try:
        # Generate audio using the worker's local pipeline
        generator = worker_pipeline(text, voice=speaker)
        
        audio_chunks = []
        for chunk_idx, (gs, ps, audio) in enumerate(generator):                
            audio_chunks.append(audio)
            
        # Safely concatenate and save if we got audio back
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            sf.write(save_path, final_audio, 24000)
            
    except Exception as e:
        print(f"Error generating {save_path}: {e}")

if __name__ == '__main__':
    # Required for PyTorch multiprocessing (prevents CUDA context crashes if using a GPU)
    mp.set_start_method('spawn', force=True)

    files = find_json('blimp')
    tasks = []

    print("Gathering tasks...")
    # 1. Flatten all tasks into a single list
    for file in files:
        items = load(file, TYPE)
        n = len(items)

        if n == 0:
            continue
        
        all_texts = [item['audio'] for item in items]
        all_speakers = generate_voice(n, "KOKORO")
        all_filenames = [item['name'] for item in items] 

        folder_name = os.path.basename(file).replace('.jsonl', '')
        output_folder = os.path.join(os.getcwd(), output_prefix, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for i in range(n):
            save_path = os.path.join(output_folder, f"{all_filenames[i].replace('.jsonl', '')}.wav")
            if not os.path.exists(save_path):
                tasks.append((all_texts[i], all_speakers[i], save_path))

    print(f"Found {len(tasks)} audio files to generate.")

    # 2. Process tasks in parallel
    # Adjust max_workers based on your CPU cores and available GPU VRAM. 
    # Start with 2 or 4. Too many will cause Out Of Memory (OOM) errors.
    WORKERS = 4 

    print(f"Starting parallel generation with {WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as executor:
        # map executes the function across all tasks in parallel
        list(executor.map(process_audio_task, tasks))
        
    print("Processing complete!")