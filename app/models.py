"""数据模型定义"""
from pydantic import BaseModel
from typing import List, Optional


class UploadResponse(BaseModel):
    status: str
    chunks: int
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[str]


class ErrorResponse(BaseModel):
    detail: str
