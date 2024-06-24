import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.qdrant import Qdrant

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

print(embeddings)

loader = DirectoryLoader(r'C:\Users\ADMIN\Desktop\LANGCHAIN\RAG PROJECTS\Medical LLM\Data',
                         glob="**/*.pdf",
                         show_progress=True,
                         loader_cls=PyPDFLoader)

documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100)
texts = text_splitter.split_documents(documents)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    path=r'C:\Users\ADMIN\Desktop\LANGCHAIN\RAG PROJECTS\Medical LLM',
    collection_name="medical_rag"
)
print("Vector DB Successfully Created!")