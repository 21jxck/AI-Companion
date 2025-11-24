from flask import Flask, request, jsonify, make_response
app = Flask(__name__)

from langchain_ollama import ChatOllama
print("Carico il modello")
lcmodel = ChatOllama(model="gemma3:4b", temperature=0, reasoning=False)
print("Modello caricato")

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
vs = InMemoryVectorStore.load("./vs/alice.db", embeddings)
retriever = vs.as_retriever()

from kokoro import KPipeline, KModel    # genera l'audio
import soundfile as sf                  # permette di gestire i file waveform
from pydub import AudioSegment          # permette di unire i file
import base64                           # permette la codifica del file come base64

kmodel = KModel(model="models/kokoro-v1_0.pth", config="models/config.json")
pipeline = KPipeline(lang_code="i", model=kmodel)

print("Creo il contesto")

context = [('system', 'sei il Cappellaio Matto che racconta la sua storia'),
           ('system', '')
          ]

lcmodel.invoke(context)

print("Contesto creato")

@app.route('/aicompanion', methods=['POST', 'OPTIONS'])
def aicompanion():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    if request.is_json:
        domanda = (request.json.get('domanda') or "").strip()
    else:
        domanda = (request.form.get('domanda') or "").strip()

    if not domanda:
        return jsonify({"error": "Nessuna domanda ricevuta"}), 400

    print(f"Domanda ricevuta: {domanda}")

    context.append(('human', domanda))

    # utilizziamo il rag
    documents = retriever.invoke(domanda)
    doc_texts = "\n".join(doc.page_content for doc in documents)
    doc_texts = "usa questo contesto per rispondere, attieniti solo a questo contesto, rispondi solo alle domande che riguardano il tuo mondo (il paese delle meraviglie) e non alle domande che richiedono qualcosa della Terra e se non ci riesci rispondi boh: " + doc_texts
    context[1] = ('system', doc_texts)

    response = lcmodel.invoke(context)
    context.append(('ai', response.content))

    print(f"Risposta AI: {response.content}")

    wav_b64 = text_to_speech(response.content)

    response = jsonify({
        "content" : str(response.content),
        "language" : "Italian",
        "base64" : wav_b64
    })

    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

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

    with open("./generated/audio.wav", "rb") as f:
        wav_bytes = f.read()

    wav_b64 = base64.b64encode(wav_bytes).decode("utf-8")

    return wav_b64

if __name__ == '__main__':
    print("Avvio del server sulla porta 9000...")
    app.run(port=9000, debug=False)
