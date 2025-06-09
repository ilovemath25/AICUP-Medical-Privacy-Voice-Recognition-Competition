from Speech2Text import model
def transcribe(audioFile):
    if not audioFile.endswith((".wav", ".mp3", ".flac")):
        return None
    result = model.transcribe(audioFile, word_timestamps=True)
    result["text"] = result["text"][0].upper() + result["text"][1:]
    segments = result.get("segments", [])
    words = []
    for segment in segments:
        for word in segment["words"]:
            words.append({
                "word": word["word"].strip('.,!?" '),
                "start": word["start"],
                "end": word["end"]
            })
    return {"text" : result["text"], "words" : words}