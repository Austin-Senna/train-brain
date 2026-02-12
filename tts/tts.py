import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os 

prompt = "Kevin Wu you're so hot."
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
)

ref_audio = "speaker.wav"
ref_text  = "People always point out that my voice is kind of, like, weird and raspy, or it sounds like I'm trying to talk louder than I really am. It's because, uh, I'd always get yelled at, especially by this one specific teacher. Everyone hates doing presentations in school, right? "

wavs, sr = model.generate_voice_clone(
    text=prompt,
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write(f"{os.getcwd()}/austin_clone.wav", wavs[0], sr)