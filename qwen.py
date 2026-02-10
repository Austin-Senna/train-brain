import os
import torch
from loader import load, audio_generator
import transformers
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# 1. Force logging
transformers.utils.logging.set_verbosity_info()

print("--- Step 1: Loading Processor ---")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

print("--- Step 2: Loading Model (This may take 2-5 mins on NFS) ---")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", 
    device_map="auto",              # Checks for your NCSA GPUs
    torch_dtype=torch.float16,      # Essential for memory/speed
    low_cpu_mem_usage=True          # Prevents NFS stalls
)

print(f"--- Step 3: Model Loaded on device. ---")
audios = load('mald1/MALD1_pw/')
subset_audios = audios[:100]

for i, (audio, file_name) in enumerate(audio_generator(subset_audios, processor=processor)):
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": file_name},
            {"type": "text", "text": "Transcribe the audio."},
        ]}
    ]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate) 
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    penultimate_embeddings = outputs.hidden_states[-2]
    save_name = os.path.basename(file_name).replace('.wav', '.pt')
    os.makedirs(f"{os.getcwd()}/output_features", exist_ok=True)
    torch.save(penultimate_embeddings.cpu(), f"output_features/{save_name}")
    print(f"Processed: {save_name} | Features Extracted.")

    generate_ids = model.generate(**inputs, max_new_tokens=512)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"response: {response}")
    