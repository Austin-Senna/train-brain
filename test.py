from loader import load, audio_generator, load_json, find_json
import os


files = find_json('blimp')
file = files[0]
print(load_json(file)[:10])
# print(files)
# audios = load('mald1/MALD1_pw/')
# subset_audios = audios[:10]
# print(subset_audios)
# print("loaded audio")

# os.makedirs(f"{os.getcwd()}/output_features", exist_ok=True)
# for i, (audio, file_name) in enumerate(audio_generator(subset_audios, processor=None)):
#     print(f'Processing file {i}')
#     print(audio, file_name)