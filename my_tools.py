import base64
import os
def image_to_base64(image_path: str):
    """
    将本地图片转换为 Base64 编码的 data URI。
    :param image_path: 本地图片路径，如 "D:/images/math.jpg"
    :return: base64
    """
    # 检查文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")
    # 读取图片并编码
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    return b64_str
   