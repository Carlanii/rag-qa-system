"""FastAPI 后端入口"""
from fastapi import FastAPI, UploadFile, File
from app.rag_engine import RAGEngine
from app.models import UploadResponse, QueryRequest, QueryResponse
from app.utils import save_upload_temp, cleanup_temp
import os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG 智能文档问答系统", description="基于 LangChain + ChromaDB 的检索增强生成系统", version="1.0.0")
engine = RAGEngine()


@app.get("/")
async def root():
    return {"message": "RAG 智能文档问答系统 API", "version": "1.0.0"}


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        return UploadResponse(status="error", chunks=0, message="文件名为空")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        return UploadResponse(status="error", chunks=0, message="仅支持 PDF 和 TXT 格式")
    try:
        data = await file.read()
        tmp = save_upload_temp(data, file.filename)
        count = engine.build_knowledge_base([tmp])
        cleanup_temp(tmp)
        logger.info(f"文档 {file.filename} 处理完成，生成 {count} 个文本块")
        return UploadResponse(status="ok", chunks=count, message=f"文档处理成功，生成 {count} 个文本块")
    except Exception as e:
        logger.error(f"上传失败: {e}")
        return UploadResponse(status="error", chunks=0, message=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    try:
        r = engine.query(req.question, k=req.top_k)
        return QueryResponse(question=r["question"], answer=r["answer"], source_documents=r["source_documents"])
    except Exception as e:
        logger.error(f"查询失败: {e}")
        return QueryResponse(question=req.question, answer=f"查询出错: {str(e)}", source_documents=[])


@app.post("/clear")
async def clear():
    try:
        engine.clear()
        return {"status": "ok", "message": "知识库已清空"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
