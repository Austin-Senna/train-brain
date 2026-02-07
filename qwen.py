import os

# 1. SET THE PATH FIRST
# project_path = "/projects/bgbh/awijaya/train-brain/models" 
# os.makedirs(project_path, exist_ok=True)

# Tell Hugging Face: "Put everything here, forever."
# os.environ["HF_HOME"] = project_path
# # 2. NOW IMPORT TRANSFORMERS
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch

# print(f"Downloading model to stable storage: {project_path} ...")

model_id = "Qwen/Qwen2-Audio-7B-Instruct"

# # 3. LOAD (This will download to /projects/bbXX/... instead of ~/.cache)
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" 
)

# print("Model successfully saved to Projects folder!")