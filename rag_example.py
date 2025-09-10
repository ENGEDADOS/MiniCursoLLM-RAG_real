import os
import getpass
import requests
import numpy as np
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

load_dotenv('example.env')

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("Faltando TAVILY_API_KEY no .env")

if not GOOGLE_API_KEY:
    raise ValueError("Faltando GOOGLE_API_KEY no .env")

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

tavily_tool = TavilySearch(max_results=5, topic="general", include_answer=True)

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

loader = PyPDFLoader("LLM_introduction.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

rag = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True
)

def normalize_vector(v):
    return v / np.linalg.norm(v)

def consulta_rag_ou_web(query: str, score_threshold: float = 10):
    docs_com_scores = vectorstore.similarity_search_with_score(query, k=4)

    # for doc, score in docs_com_scores:
    #     print(f'Doc: "{doc.page_content[:200]}"... Score: {score}\n')

    docs_relevantes = [doc for doc, score in docs_com_scores if score <= score_threshold]

    if not docs_relevantes:
        print('\n- Buscando Informações Relevantes na Web')
        buscador = tavily_tool.invoke(query)
        resposta = buscador.get("answer") or "Não encontrei resposta local, usando web search."
        fontes = ["Web: " + "; ".join(r["url"] for r in buscador.get("results", []))]
        return resposta, fontes

    resultado = rag.invoke(query)
    resposta = resultado["result"]
    fontes = [
        doc.metadata.get("source", "") or doc.page_content[:200]
        for doc in resultado["source_documents"]
    ]

    if len(resposta.strip()) < 50:
        buscador = tavily_tool.invoke(query)
        resposta = buscador.get("answer") or resposta
        fontes.append("Web: " + "; ".join(r["url"] for r in buscador.get("results", [])))

    return resposta, fontes

if __name__ == "__main__":
    instructions = """
--------------------------
COMANDOS:
/quit -> encerrar programa
--------------------------

"""
    print(instructions)
    while True:
        pergunta = input('Digite algo: ')
        if (pergunta.strip() == '/quit'):
            break
        resposta, fontes = consulta_rag_ou_web(pergunta, 10)
        print("\nResposta:", resposta)
        print("\nFontes:")
        for f in fontes:
            print("-", f)
