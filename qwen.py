import os
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import torch
import io
# project_path = "/projects/bgbh/awijaya/train-brain/models" 
# os.makedirs(project_path, exist_ok=True)
# os.environ["HF_HOME"] = project_path

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="auto")

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20260207%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260207T021823Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=9993a208cae876e1c63988d36dfd42169c32882f61d41430fafcb6a9566a70a8574f06c60cc1fb8bb291377ba26da7d231dd23260b6a49ba0764b19f91a30dccb2930dc217d3f67d37c2c4cc5ff88e3daade20013372441ee8bceff0b1f895cb309f53bc0d113317f40f504a3c904bb94e9974f643bf573bcbadc951a9a8da3e606f44951071915b64ead25b5d327e4ab6bccb2483a6fb14d07dae48083ea58f46300744adf04df953fb31640669c28692a24a9826c126e4925116ef3d4acabd981dab7180626e3378055bd012b08a1a780d6ff5b7f9a0d0d18b52482ede55f5b139e7fedb6e08cd1579f2fc711668ff6397429db8e21427027beeb4cec42f8f"},
        {"type": "text", "text": "Transcribe the audio."},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                audios.append(
                    librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()), 
                        sr=processor.feature_extractor.sampling_rate)[0]
                )

inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True) 
inputs = inputs.to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=512)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"response: {response}")