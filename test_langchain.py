from langchain_ollama import ChatOllama
#from langchain_ollama.llms import OllamaLLM

print("Carico il modello")
#model = OllamaLLM(model="gemma3:4b")
lcmodel = ChatOllama(model="gemma3:4b", temperature=0, reasoning=False)
print("Modello caricato")

print("Creo il contesto")
context = [
           ('system', 'sei un professore di astrofisica dell\'università di Padova. Tieni lezioni a gruppi di 100 studenti in una grande aula. Rispondi in non più di 100 parole.')
          ]
lcmodel.invoke(context)
print("Contesto creato")

context.append(('human', 'Che cos\'è la luna?'))
response = lcmodel.invoke(context)

print(response.content)
context.append(('ai', response.content))

context.append(('human', 'Qualche altro esempio di satelliti nel sistema solare?'))
response = lcmodel.invoke(context)

print(response.content)
context.append(('ai', response.content))