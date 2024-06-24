import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma

model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loader = DirectoryLoader(r'C:\Users\ADMIN\Desktop\LANGCHAIN\RAG_PROJECTS\Investment_Bank_LLM\Data',
                         glob="**/*.pdf",
                         show_progress=True,
                         loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

vector_store = Chroma.from_documents(texts[:50],
                                     embeddings,
                                     collection_metadata={"hnsw:space": "cosine"},
                                     persist_directory="RAG_PROJECTS/Investment_Bank_LLM/stores/investment_bank")

print("Vector Store Created.......")