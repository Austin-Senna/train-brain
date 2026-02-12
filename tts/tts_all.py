from loader import find_json, load_json, generate_voice
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os 


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
files = find_json('blimp')

for i, file in enumerate(files):
    print(f"Processing: {file}")
    items = load_json(file)
    n = len(items)

    if n == 0:
        continue

    wavs, sr = model.generate_custom_voice(
    text=[item['audio'] for item in items],
    language=["English"] * n,
    speaker=generate_voice(n),
    instruct=[""] * n
    )

    for item, wav in zip(items, wavs):
        filename = item['name']
        save_path = os.path.join(os.getcwd(), f"{filename}.wav")
        
        if hasattr(wav, 'cpu'):
            wav = wav.cpu().numpy()
            
        sf.write(save_path, wav, sr)


