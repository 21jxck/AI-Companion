import whisper

wmodel = whisper.load_model(name="./models/small.pt", device="cpu") #se installi cuda puoi mettere su device "cuda"

def speech_to_text(audio_path) :
    result = wmodel.transcribe(audio=audio_path, language="it", fp16=False)
    return result["text"]

print(speech_to_text("C:/Users/jacopo.olivo/aicompanion/audio.wav"))
