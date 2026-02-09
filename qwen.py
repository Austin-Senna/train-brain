import os
from io import BytesIO
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import torch
from loader import load, audio_generator
# import io

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

audios = load('mald1/MALD1_pw/')

for i, (audio, file_name) in audio_generator(audios, processor=processor):
    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio_url": file_name},
            {"type": "text", "text": "Transcribe the audio."},
        ]}
    ]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True) 
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    penultimate_embeddings = outputs.hidden_states[-2]
    save_name = os.path.basename(file_name).replace('.wav', '.pt')
    torch.save(penultimate_embeddings.cpu(), f"output_features/{save_name}")
    print(f"Processed: {save_name} | Features Extracted.")

    generate_ids = model.generate(**inputs, max_new_tokens=512)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"response: {response}")