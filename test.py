from loader import load, audio_generator

audios = load('mald1/MALD1_pw/')
subset_audios = audios[:10]
print("loaded audio")

for i, (audio, file_name) in audio_generator(subset_audios, processor=None):
    print('in librosa')
    print(audio, file_name)