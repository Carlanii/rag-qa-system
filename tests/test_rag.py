"""RAG 引擎测试"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.rag_engine import RAGEngine
from app.config import settings


def test_engine_init():
    engine = RAGEngine(persist_dir="./test_chroma")
    assert engine is not None
    engine.clear()


def test_load_document(tmp_path):
    engine = RAGEngine(persist_dir="./test_chroma")
    test_file = tmp_path / "test.txt"
    test_file.write_text("这是一段测试文本，用于验证 RAG 系统的文档加载功能。")
    docs = engine._load(str(test_file))
    assert len(docs) > 0
    engine.clear()


def test_build_and_query(tmp_path):
    engine = RAGEngine(persist_dir="./test_chroma")
    test_file = tmp_path / "test.txt"
    test_file.write_text("Python 是一种广泛使用的编程语言。它由 Guido van Rossum 创建。")
    count = engine.build_knowledge_base([str(test_file)])
    assert count > 0
    engine.clear()


def test_invalid_provider():
    old = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "invalid"
    try:
        from app.rag_engine import get_llm
        try:
            get_llm()
            assert False
        except ValueError:
            assert True
    finally:
        settings.LLM_PROVIDER = old
