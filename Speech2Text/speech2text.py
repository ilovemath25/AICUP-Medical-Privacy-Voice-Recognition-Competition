from Speech2Text import model
def transcribe(audioFile):
    if not audioFile.endswith((".wav", ".mp3", ".flac")):
        return None
    result = model.transcribe(audioFile)
    result["text"] = result["text"][0].upper() + result["text"][1:]
    return result["text"]

def transcribe_with_timestamps(audioFile):
    if not audioFile.endswith((".wav", ".mp3", ".flac")):
        return None
    result = model.transcribe(audioFile, word_timestamps=True)
    segments = result.get("segments", [])
    results = []
    for segment in segments:
        tempSegment = []
        for s in segment["words"]:
            tempSegment.append({
                "word": s["word"],
                "start": s["start"],
                "end": s["end"]
            })
        results.append(tempSegment)
    return results