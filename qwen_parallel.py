import os
import torch
import logging
import sys
from loader import load, audio_generator
import transformers
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import librosa

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
BATCH_SIZE = 4 

# ... [Keep your Step 3: DATA LOADING exactly the same] ...

logger.info(f"Found {len(audios)} files to process.")
os.makedirs(output_folder, exist_ok=True)

# 4. BATCHED EXTRACTION LOOP
# We chunk the list of files into batches
for i in range(0, len(audios), BATCH_SIZE):
    batch_files = audios[i : i + BATCH_SIZE]
    
    try:
        texts = []
        loaded_audios = []
        valid_files = [] # Keep track in case one audio file is corrupted and fails to load
        
        # 1. Prepare the batch
        for file_name in batch_files:
            try:
                # Load audio data (assuming librosa or your loader, keeping it generic)
                # You might need to adjust this depending on how your `loader` works
                audio_array, _ = librosa.load(file_name, sr=processor.feature_extractor.sampling_rate)
                loaded_audios.append(audio_array)
                valid_files.append(file_name)
                
                conversation = [
                    {'role': 'system', 'content': ''}, 
                    {"role": "user", "content": [
                        {"type": "audio", "audio_url": file_name},
                        {"type": "text", "text": prompt}, 
                    ]}
                ]
                text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                texts.append(text)
            except Exception as e:
                logger.error(f"Failed to load audio {file_name}: {e}")
                
        if not valid_files:
            continue # Skip if all files in the batch failed to load
            
        # 2. Process the batch
        inputs = processor(
            text=texts, 
            audio=loaded_audios, 
            return_tensors="pt", 
            padding=True, 
            sampling_rate=processor.feature_extractor.sampling_rate
        ).to(model.device)
        
        # 3. Forward Pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        penultimate_embeddings = outputs.hidden_states[-2].cpu()
        
        # 4. Save each item in the batch
        for batch_idx, file_name in enumerate(valid_files):
            base_name = os.path.basename(file_name)
            save_name = base_name.replace('.wav', '.pt')
            
            if FOLDER_MODE == "CREATE":
                name_without_ext = base_name.replace('.wav', '')
                subfolder, index, verdict = name_without_ext.rsplit('_', 2) 
                save_folder = os.path.join(output_folder, subfolder, verdict)
                os.makedirs(save_folder, exist_ok=True)
            else:
                save_folder = output_folder
                
            save_path = os.path.join(save_folder, save_name)
            
            # Extract the specific embedding for this item in the batch
            # Note: Because of padding, the sequence length might be longer than the actual audio.
            # If you need strict exact sequence lengths without padding, batching becomes much more complex.
            single_embedding = penultimate_embeddings[batch_idx] 
            
            torch.save(single_embedding, save_path)
            
        logger.info(f"[{min(i + BATCH_SIZE, len(audios))}/{len(audios)}] Processed batch of {len(valid_files)}")

        # Free up memory explicitly
        del outputs, inputs, penultimate_embeddings, single_embedding
        torch.cuda.empty_cache() # Helpful when batching to prevent fragmentation

    except Exception as e:
        logger.error(f"Failed to process batch starting with {batch_files[0]}: {e}")
        continue