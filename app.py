from Speech2Text.speech2text import transcribe
from NER_Shi.ner_shi import predictTag
import os

a = "Training_Dataset_01"
b = "Training_Dataset_02"
audio_folder = "./audio_folder/" + a + "/audio"
results1 = []
results2 = []

def process_file(filename):
    audio_path = os.path.join(audio_folder, filename)
    result = transcribe(audio_path)
    # Task 1
    results1.append({"file": filename, "transcription": result["text"]})
    print(result["text"])
    # Task 2
    tagResult = predictTag(result["text"])
    tempResult = result["words"].copy()
    for i in range(len(tagResult)):
        for j in range(len(tempResult)):
            word1 = tagResult[i]["word"]
            word2 = tempResult[j]["word"]
            if word1 == word2:
                tagResult[i]["start"] = tempResult[j]["start"]
                tagResult[i]["end"] = tempResult[j]["end"]
                tempResult[j]["word"] = ""
                break
    for tag in tagResult:
        results2.append({
            "file": filename,
            "tag": tag["entity_group"],
            "start": tag["start"],
            "end": tag["end"],
            "word": tag["word"]
        })

for filename in os.listdir(audio_folder):
    process_file(filename)
with open('output1.txt', 'w') as file:
    for result in results1:
        file.write(result["file"] + '\t' + result["transcription"] + '\n')
with open('output2.txt', 'w') as file:
    for result in results2:
        file.write(result["file"] + '\t' + result["tag"] + '\t' + str(result["start"]) + '\t' + str(result["end"]) + '\t' + result["word"] + '\n')