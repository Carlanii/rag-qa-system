"""RAG核心引擎"""
import os
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from app.config import settings


def get_llm():
    p = settings.LLM_PROVIDER
    if p == "openai":
        return ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.3, openai_api_key=settings.OPENAI_API_KEY)
    elif p == "deepseek":
        from langchain_openai import ChatOpenAI as DS
        return DS(model=settings.DEEPSEEK_MODEL, temperature=0.3, openai_api_key=settings.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    elif p == "qwen":
        from langchain_openai import ChatOpenAI as QW
        return QW(model=settings.QWEN_MODEL, temperature=0.3, openai_api_key=settings.QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    raise ValueError(f"unsupported provider: {p}")


def get_embeddings():
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY)


class RAGEngine:
    def __init__(self, persist_dir=None):
        self.persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        self.vector_store = None
        os.makedirs(self.persist_dir, exist_ok=True)

    def build_knowledge_base(self, file_paths):
        docs = []
        for p in file_paths:
            docs.extend(self._load(p))
        if not docs:
            raise ValueError("no documents loaded")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory=self.persist_dir)
        self.vector_store.persist()
        return len(chunks)

    def _load(self, path):
        if not os.path.exists(path):
            return []
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return PyPDFLoader(path).load()
        elif ext == ".txt":
            return TextLoader(path, encoding="utf-8").load()
        return []

    def query(self, question, k=3):
        if not self.vector_store:
            self.vector_store = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是智能文档问答助手，基于上下文回答问题。找不到就说不知道。\n上下文：{context}"),
            ("human", "{input}"),
        ])
        chain = create_retrieval_chain(retriever, create_stuff_documents_chain(self.llm, prompt))
        r = chain.invoke({"input": question})
        sources = list(set(d.metadata.get("source", "") for d in r.get("context", []) if d.metadata.get("source")))
        return {"question": question, "answer": r["answer"], "source_documents": sources}

    def clear(self):
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
        self.vector_store = None
