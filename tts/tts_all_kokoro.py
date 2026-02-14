from loader import find_json, load_json, generate_voice
from kokoro import KPipeline
import soundfile as sf
import torch
import os 

TYPE = "SEMANTIC"
output_prefix = "BLIMP_KOKORO"
pipeline = KPipeline(lang_code='a')
files = find_json('blimp')

for file in files:
    print(f"Processing: {file}")
    items = load_json(file, TYPE)
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
            generator = pipeline(all_texts[i], voice=all_speakers[i])
            for chunk_idx, (gs, ps, audio) in enumerate(generator):                
                sf.write(save_path, audio, 24000)