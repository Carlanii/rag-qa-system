"""工具函数"""
import os, tempfile

def save_upload_temp(file_data, filename):
    """保存上传文件到临时目录"""
    tmp_dir = tempfile.mkdtemp()
    safe_name = os.path.basename(filename) if filename else "uploaded_file.txt"
    file_path = os.path.join(tmp_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(file_data)
    return file_path

def cleanup_temp(file_path):
    """清理临时文件"""
    if os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except:
            pass
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path):
            try:
                os.rmdir(dir_path)
            except:
                pass
