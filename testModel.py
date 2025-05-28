from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import librosa
import pandas as pd
import os

model_path = "./wav2vec2_finetuned"
processor = Wav2Vec2Processor.from_pretrained(model_path)
model = Wav2Vec2ForCTC.from_pretrained(model_path)

audio_folder = "./audio_dataset/MedicalSpeechIntent"
results = []
max = 10
count = 0

for filename in os.listdir(audio_folder):
    if count >= max:
        break
    if filename.lower().endswith((".wav", ".mp3", ".flac")):
        audio_path = os.path.join(audio_folder, filename)
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            y_trimmed, _ = librosa.effects.trim(y)
            input_values = processor(y_trimmed, return_tensors="pt", sampling_rate=sr).input_values
            with torch.no_grad():
                logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


        except Exception as e:
            transcription = f"Error: {e}"
        results.append({"file": filename, "transcription": transcription})
        print(transcription)
        count += 1