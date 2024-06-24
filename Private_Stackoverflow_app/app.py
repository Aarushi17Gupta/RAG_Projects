from langchain_community.llms.ctransformers import CTransformers
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
import json
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from fastapi import FastAPI, Form, Request, Response
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn

app = FastAPI()

templates = Jinja2Templates(directory=r"C:\Users\ADMIN\Desktop\LANGCHAIN\RAG_PROJECTS\Private_Stackoverflow_app\templates")

DB_FAISS_PATH = r'C:\Users\ADMIN\Desktop\LANGCHAIN\RAG_PROJECTS\Private_Stackoverflow_app\vectorstore\db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Loading the model
def load_llm():
    llm = CTransformers(
        model = "TheBloke/stablecode-instruct-alpha-3b-GGML",
        model_type="gpt_neox",
        max_new_tokens = 512,
        temperature = 0.7
    )
    return llm

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='microsoft/unixcoder-base',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(request: Request, query: str = Form(...)):
    resp = final_result(query)    
    result = resp['result']
    print(resp)
    for i in resp['source_documents'][0]:
        if 'metadata' in i:
            source_doc = i[1]['source']
    response_data = jsonable_encoder(json.dumps({"result": result, "source_doc": source_doc}))
    res = Response(response_data)
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)