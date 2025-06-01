from Speech2Text import model
def transcribe(audioFile):
    if not audioFile.endswith([".wav", ".mp3", ".flac"]):
        return None
    result = model.transcribe(audioFile)
    result["text"] = result["text"][0].upper() + result["text"][1:]
    return result["text"]

