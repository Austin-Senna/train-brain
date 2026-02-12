import os
import torch
import logging
import sys
from loader import load, audio_generator
import transformers
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

output_folder = "output_features_02/rw"
audio_folder = 'mald1/MALD1_rw/'
output_run_log = "02_rw"
# prompt = "Output the transcription of the audio only."
prompt = "Listen to the following word. Is it a real English word or a pseudoword? Answer with 'Real' or 'Pseudo'."

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{output_run_log}_extraction_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 2. MODEL LOADING
logger.info("--- Step 1: Loading Processor ---")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

logger.info("--- Step 2: Loading Model (This may take 2-5 mins on NFS) ---")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", 
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
logger.info(f"--- Step 3: Model Loaded on device {model.device} ---")

# 3. DATA LOADING
audios = load(audio_folder)
audios_to_run = []
for f in audios:
    expected_output = os.path.join(output_folder, os.path.basename(f).replace('.wav', '.pt'))
    if not os.path.exists(expected_output):
        audios_to_run.append(f)
audios = audios_to_run

logger.info(f"Found {len(audios)} files to process.")

os.makedirs(output_folder, exist_ok=True)

# 4. EXTRACTION LOOP
for i, (audio, file_name) in enumerate(audio_generator(audios, processor=processor)):
    try:
        # Prepare Inputs
        conversation = [
            {'role': 'system', 'content': ''}, 
            {"role": "user", "content": [
                {"type": "audio", "audio_url": file_name},
                {"type": "text", "text": prompt}, 
            ]}
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            text=text, 
            audio=audio, 
            return_tensors="pt", 
            padding=True, 
            sampling_rate=processor.feature_extractor.sampling_rate
        ).to(model.device)
        
        # Forward Pass (Extract Features)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        penultimate_embeddings = outputs.hidden_states[-2]
        
        # Save
        save_name = os.path.basename(file_name).replace('.wav', '.pt')
        save_path = os.path.join(output_folder, save_name)
        torch.save(penultimate_embeddings.cpu(), save_path)
        
        logger.info(f"[{i}/{len(audios)}] Processed: {save_name}")

        # Generation (Optional - slow!)
        # generate_ids = model.generate(**inputs, max_new_tokens=512)
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        # logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Failed to process file {file_name}: {e}")
        continue # Skip to next file so the job doesn't die