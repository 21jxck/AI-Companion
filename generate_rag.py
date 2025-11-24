from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore

# carico il documento
loader = PyPDFLoader("vs/data/carroll_alice_nel_etc_loescher.pdf")
docs = loader.load()
print("Documenti caricati", len(docs))

# divido il documento in parti piu piccole
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 500,
    length_function = len,
    is_separator_regex = False
)
chunks = text_splitter.split_documents(docs)
print("Chunk caricati", len(chunks))

# creo i vettori
embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
vs = InMemoryVectorStore.from_documents(chunks, embeddings)
print("Vector Store creato")

# salvo tutto su file
vs.dump("./vs/alice.db")
print("Vector Store salvato")