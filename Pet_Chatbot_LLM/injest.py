import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = PyPDFLoader("RAG_PROJECTS/Pet_Chatbot_LLM/pet.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(texts,
                                     embeddings,
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory="RAG_PROJECTS/Pet_Chatbot_LLM/stores/pet_cosine")

print("Vector Store Created.......")