from Speech2Text.speech2text import transcribe, transcribe_with_timestamps
import os

audio_folder = "./audio_dataset/AudioFromCompetition"
results = []
max = 1
count = 0
# Task 1
for filename in os.listdir(audio_folder):
    if (count >= max): 
        break
    audio_path = os.path.join(audio_folder, filename)
    result = transcribe(audio_path)
    # results.append({"file": filename, "transcription": result})
    print(result)
    count += 1

# Task 2
count = 0
for filename in os.listdir(audio_folder):
    if (count >= max): 
        break
    audio_path = os.path.join(audio_folder, filename)
    result = transcribe_with_timestamps(audio_path)
    # results.append({"file": filename, "transcription": result})
    for r in result: print(r)
    count += 1