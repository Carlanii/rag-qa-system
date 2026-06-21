# RAG 智能文档问答系统

> 基于 LangChain + ChromaDB + FastAPI 构建的检索增强生成（RAG）问答系统

## 项目概述

本项目实现了一个完整的 RAG（Retrieval-Augmented Generation）智能问答系统，支持上传 PDF/TXT 文档构建知识库，并基于文档内容进行智能问答。采用前后端分离架构，后端提供 RESTful API，前端使用 Streamlit 构建交互界面。

### 核心功能

- **文档知识库构建**：上传 PDF/TXT 文档，自动完成文档解析、文本分割、向量化存储
- **智能问答**：基于文档内容的语义检索 + LLM 生成，回答有据可依
- **多 LLM 后端**：支持 OpenAI、DeepSeek、阿里 Qwen 三种大模型

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI (Python) |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| LLM 后端 | OpenAI GPT / DeepSeek / Qwen |
| Embedding | text-embedding-3-small |
| 前端界面 | Streamlit |

## 项目结构

```
rag-qa-system/
├── app/
│   ├── main.py          # FastAPI 入口 & API 路由
│   ├── rag_engine.py    # RAG 核心引擎（检索 + 生成）
│   ├── config.py        # 配置管理（多 LLM 提供商切换）
│   ├── models.py        # Pydantic 数据模型
│   └── utils.py         # 工具函数
├── ui/
│   └── streamlit_app.py # Streamlit 交互界面
├── tests/
│   └── test_rag.py
├── docs/sample_docs/    # 示例文档
├── scripts/setup.bat    # Windows 安装脚本
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

### 1. 前置条件
- Python 3.9+
- 一个 LLM API Key（三选一）：
  - **DeepSeek**（推荐，国内直连，便宜）：https://platform.deepseek.com/api_keys
  - **OpenAI**：https://platform.openai.com/api-keys
  - **阿里 Qwen**（有免费额度）：https://dashscope.aliyun.com/

### 2. 安装

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install streamlit uvicorn python-dotenv

# 配置环境变量
copy .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 配置 API Key

编辑 `.env` 文件：

```
# 推荐使用 DeepSeek（国内直连）
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-deepseek-api-key

# 或使用 OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-openai-api-key
```

### 4. 启动

终端 1（后端）：
```bash
uvicorn app.main:app --reload --port 8000
```

终端 2（前端）：
```bash
streamlit run ui/streamlit_app.py
```

浏览器打开 http://localhost:8501 即可使用。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| / | GET | 健康检查 |
| /upload | POST | 上传文档并构建知识库 |
| /query | POST | 问答查询 |
| /clear | POST | 清空知识库 |

## 面试核心知识点

### RAG 核心流程
用户提问 -> 语义检索（向量相似度）-> 上下文增强 -> LLM 生成 -> 返回答案

### 为什么用 RAG 而不是直接问 LLM？
- 可以基于私有知识库回答
- 减少幻觉，回答可溯源
- 知识实时更新，无需重新训练

### 关键模块
1. **文档分块**：RecursiveCharacterTextSplitter，chunk_size=500，chunk_overlap=50
2. **向量检索**：Embedding 模型将文本转向量，余弦相似度检索 Top-K
3. **Prompt 组装**：检索结果 + 用户问题 -> LLM

## License

MIT
