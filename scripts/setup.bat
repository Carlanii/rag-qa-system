@echo off
echo ====================================
echo RAG 智能文档问答系统 - 环境安装
echo ====================================

echo [1/3] 创建虚拟环境...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] 安装依赖...
pip install -r requirements.txt
pip install streamlit uvicorn python-dotenv

echo [3/3] 复制环境变量配置...
if not exist .env (
    copy .env.example .env
    echo 请编辑 .env 文件填入你的 API Key
)

echo.
echo 安装完成！
echo.
echo 启动后端: uvicorn app.main:app --reload --port 8000
echo 启动前端: streamlit run ui/streamlit_app.py
pause
