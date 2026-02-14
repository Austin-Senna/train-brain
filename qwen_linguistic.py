import os
import torch
import logging
import sys
from loader import load, audio_generator
import transformers
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# CHANGE MODE HERE:
EVALUATION_TYPE = "02" #01/02
RUN_TYPE = "syntax" #lexicon/syntax/semantics
FOLDER_MODE = "CREATE" # CREATE -> creates subfolder based on name, #LET -> default

PROMPTS = {"lexicon": 
           {
               "01": "Output the transcription of the audio only.", 
               "02": "Listen to the following word. Is it a real English word or a pseudoword? Answer with 'Real' or 'Pseudo'."
            },
           "syntax": {
               "01": "Output the transcription of the audio only.", 
               "02": "Listen to the following sentence. Is it grammatically correct or ungrammatical? Answer with 'Correct' or 'Incorrect'."
           },
           "semantics": {
               "01": "Output the transcription of the audio only.", 
               "02": "Listen to the following sentence. Does it make logical sense, or is it semantically anomalous? Answer with 'Logical' or 'Nonsense'."
           }
           }

AUDIO_FOLDERS = {"lexicon": "mald1",
                 "syntax": "BLIMP_KOKORO",
                 "semantics": ""
                 }

# CHANGE MANUALLY HERE:
output_folder = f"output_features_{EVALUATION_TYPE}/{RUN_TYPE}"
audio_folder = AUDIO_FOLDERS[RUN_TYPE]
output_run_log = f"{EVALUATION_TYPE}_{RUN_TYPE}"
prompt = PROMPTS[RUN_TYPE][str(EVALUATION_TYPE)]

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
    expected_output = ""
    base_name = os.path.basename(f)
    
    if FOLDER_MODE == "CREATE":
        name_without_ext = base_name.replace('.wav', '')
        subfolder, index, verdict = name_without_ext.rsplit('_', 2) # <-- NEW
        expected_output = os.path.join(output_folder, subfolder, verdict, f"{name_without_ext}.pt")
    else:
        expected_output = os.path.join(output_folder, base_name.replace('.wav', '.pt'))
        
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

        base_name = os.path.basename(file_name)
        save_name = base_name.replace('.wav', '.pt')
        save_folder = output_folder
        
        if FOLDER_MODE == "CREATE":
            name_without_ext = base_name.replace('.wav', '')
            # Splits from the right, ensuring prefixes with underscores stay whole
            subfolder, index, verdict = name_without_ext.rsplit('_', 2) # <-- NEW
            save_folder = os.path.join(output_folder, subfolder, verdict)
            os.makedirs(save_folder, exist_ok=True)
            
        save_path = os.path.join(save_folder, save_name)
        
        # Free up memory explicitly before saving
        embeddings_cpu = penultimate_embeddings.cpu()
        del outputs, inputs, penultimate_embeddings
        
        torch.save(embeddings_cpu, save_path)
        logger.info(f"[{i+1}/{len(audios)}] Processed: {save_name}")

        # Generation (Optional - slow!)
        # generate_ids = model.generate(**inputs, max_new_tokens=512)
        # response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        # logger.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Failed to process file {file_name}: {e}")
        continue 