"""
配置管理模块 - 支持 OpenAI / DeepSeek / Qwen 多 LLM 提供商切换
"""
import os
from dotenv import load_dotenv

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


class Settings:
    """应用配置"""

    # LLM 提供商: openai / deepseek / qwen
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER") or "openai"

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # DeepSeek
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY") or ""
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Qwen (阿里通义千问)
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-plus")

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    # 服务
    FASTAPI_HOST: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    FASTAPI_PORT: int = int(os.getenv("FASTAPI_PORT", "8000"))


settings = Settings()
