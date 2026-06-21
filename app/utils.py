"""工具函数"""
import os
import tempfile


def save_upload_temp(file_data: bytes, filename: str) -> str:
    """保存上传文件到临时目录"""
    tmp_dir = tempfile.mkdtemp()
    file_path = os.path.join(tmp_dir, filename)
    with open(file_path, "wb") as f:
        f.write(file_data)
    return file_path


def cleanup_temp(file_path: str):
    """清理临时文件"""
    if os.path.exists(file_path):
        os.unlink(file_path)
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            os.rmdir(dir_path)
