from kokoro import KPipeline, KModel    # genera l'audio
import soundfile as sf                  # permette di gestire i file waveform
from pydub import AudioSegment          # permette di unire i file
import base64                           # permette la codifica del file come base64

kmodel = KModel(model="models/kokoro-v1_0.pth", config="models/config.json")
pipeline = KPipeline(lang_code="i", model=kmodel)

def text_to_speech(text):
    generator = pipeline(text, voice='models/voices/am_fenrir.pt', speed=0.9)

    length = 0
    for i, (gs, ps, audio) in enumerate(generator) :
        sf.write(f'./generated/{i}.wav', audio, 24000, "PCM_16")
        length += 1

    sound = AudioSegment.from_wav("./generated/0.wav")
    for i in range(length) :
        if (i != 0) :
            sound = sound.append(AudioSegment.from_wav(f'./generated/{i}.wav', 0))
    sound.export("./generated/audio.wav")
                                            # scarica da nvidia cuda toolkit
text_to_speech("DIEGO ARMANDO MARADONA è IL MIGLIOR CALCIATORE DELLA STORIA!")