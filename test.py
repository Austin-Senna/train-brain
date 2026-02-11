from loader import load, audio_generator
import os
audios = load('mald1/MALD1_pw/')
subset_audios = audios[:10]
print(subset_audios)
print("loaded audio")

# os.makedirs(f"{os.getcwd()}/output_features", exist_ok=True)
# for i, (audio, file_name) in enumerate(audio_generator(subset_audios, processor=None)):
#     print(f'Processing file {i}')
#     print(audio, file_name)