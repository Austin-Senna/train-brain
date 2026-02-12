from loader import find_json, load_json, generate_voice
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os 

output_prefix = "BLIMP"
BATCH_SIZE = 4
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    attn_implementation="sdpa",
)
files = find_json('blimp')[:1]

for i, file in enumerate(files):
    print(f"Processing: {file}")
    items = load_json(file)
    n = len(items)

    if n == 0:
        continue
    
    all_texts = [item['audio'] for item in items]
    all_speakers = generate_voice(n)
    all_filenames = [item['name'] for item in items] # Assuming 'name' exists

    # Process in chunks (Batches)
    for i in range(0, n, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, n)
        print(f"  Generating batch {i} to {batch_end} of {n}...")
        
        # Slice the inputs
        batch_texts = all_texts[i : batch_end]
        batch_speakers = all_speakers[i : batch_end]
        batch_items = items[i : batch_end]

        try:
            wavs, sr = model.generate_custom_voice(
                text=batch_texts,
                language=["English"] * len(batch_texts),
                speaker=batch_speakers,
                instruct=[""] * len(batch_texts)
            )

            # Save immediately per batch
            for idx, (item, wav) in enumerate(zip(batch_items, wavs)):
                filename = item['name']
                # Create folder structure
                output_folder = os.path.join(os.getcwd(), output_prefix, os.path.basename(file).replace('.json', ''))
                os.makedirs(output_folder, exist_ok=True)
                
                # FIXED: Added 'f' before the string for formatting
                save_path = os.path.join(output_folder, f"{filename}.wav")
                
                if hasattr(wav, 'cpu'):
                    wav = wav.cpu().numpy()
                    
                sf.write(save_path, wav, sr)

        except torch.OutOfMemoryError:
            print("  ! OOM Error in batch. Try lowering BATCH_SIZE.")
            torch.cuda.empty_cache()
            break # Or implement retry logic
            
        # Optional: explicit cleanup to prevent fragmentation
        del wavs
        torch.cuda.empty_cache()
        
    del wavs
    torch.cuda.empty_cache()
    # wavs, sr = model.generate_custom_voice(
    #     text=[item['audio'] for item in items],
    #     language=["English"] * n,
    #     speaker=generate_voice(n),
    #     instruct=[""] * n
    # )

    # for item, wav in zip(items, wavs):
    #     filename = item['name']
    #     output_folder = os.path.join(os.getcwd(), output_prefix, os.path.basename(file))
    #     os.makedirs(output_folder, exist_ok=True)
    #     save_path = os.path.join(output_folder,"{filename}.wav")
        
    #     if hasattr(wav, 'cpu'):
    #         wav = wav.cpu().numpy()
            
    #     sf.write(save_path, wav, sr)


