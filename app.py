from Speech2Text.speech2text import transcribe
from NER_Shi.ner_shi import predict
import os

audio_folder = "./audio_dataset/AudioFromCompetition"
results1 = []
results2 = []
max = 1
count = 0
for filename in os.listdir(audio_folder):
    if (count >= max): 
        break
    audio_path = os.path.join(audio_folder, filename)
    result = transcribe(audio_path)
    # Task 1
    results1.append({"file": filename, "transcription": result["text"]})
    print(result["text"])
    # Task 2
    tagResult = predict(result["text"])
    count += 1