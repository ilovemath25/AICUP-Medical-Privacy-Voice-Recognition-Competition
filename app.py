from Speech2Text.speech2text import transcribe
import pandas as pd
import os

audio_folder = "./audio_dataset/AudioFromCompetition"
results = []

for filename in os.listdir(audio_folder):
    audio_path = os.path.join(audio_folder, filename)
    result = transcribe(audio_path)
    results.append({"file": filename, "transcription": result})
    print(result)
df = pd.DataFrame(results)
df.to_csv("transcriptions.csv", index=False)