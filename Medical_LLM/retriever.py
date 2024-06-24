from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")


client = QdrantClient(
    path = r'C:\Users\ADMIN\Desktop\LANGCHAIN\RAG PROJECTS\Medical LLM'
)

print(client)
print("##############")

db = Qdrant(client=client,
            embeddings=embeddings,
            collection_name="medical_rag")

print(db)
print("######")
query = "What is Metastatic disease?"

docs = db.similarity_search_with_score(query=query, k=2)
for i in docs:
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})