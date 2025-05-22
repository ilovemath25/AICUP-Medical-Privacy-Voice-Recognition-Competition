from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import pandas as pd
import os

processor = Wav2Vec2Processor.from_pretrained("./wav2vec2_local")
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2_local")

audio_folder = "./audio"
results = []

for filename in os.listdir(audio_folder):
    if filename.lower().endswith((".wav", ".mp3", ".flac")):
        audio_path = os.path.join(audio_folder, filename)
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            y_trimmed, _ = librosa.effects.trim(y)
            input_values = processor(y_trimmed, return_tensors="pt", sampling_rate=sr).input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
        except Exception as e:
            transcription = f"Error: {e}"
        results.append({"file": filename, "transcription": transcription})
        print(transcription)
df = pd.DataFrame(results)
df.to_csv("transcriptions.csv", index=False)
print("Saved transcriptions to transcriptions.csv")