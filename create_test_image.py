#!/usr/bin/env python3
"""
创建包含中文文本的测试图片
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_chinese_text_image():
    """创建包含中文文本的图片"""
    # 创建PIL图片
    width, height = 1080, 480
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # 尝试使用系统字体
    font_size = 40
    font_paths = [
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        '/System/Library/Fonts/Helvetica.ttc',  # macOS fallback
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
        'C:/Windows/Fonts/simhei.ttf',  # Windows
    ]

    font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue

    if font is None:
        # 使用默认字体
        font = ImageFont.load_default()

    # 添加中文文本
    text = "可他是我孙子呀"

    # 计算文本位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # 绘制文本
    draw.text((x, y), text, fill='black', font=font)

    # 转换为OpenCV格式并保存
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 保存图片
    output_path = 'examples/frame_000182.jpg'
    cv2.imwrite(output_path, cv_img)
    print(f"创建测试图片: {output_path}")

    # 创建第二张图片
    img2 = Image.new('RGB', (width, height), color='white')
    draw2 = ImageDraw.Draw(img2)

    text2 = "你就是个吃我们家喝我们家的赘婿"
    bbox2 = draw2.textbbox((0, 0), text2, font=font)
    text_width2 = bbox2[2] - bbox2[0]
    text_height2 = bbox2[3] - bbox2[1]

    x2 = (width - text_width2) // 2
    y2 = (height - text_height2) // 2

    draw2.text((x2, y2), text2, fill='black', font=font)

    cv_img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    output_path2 = 'examples/frame_001308.jpg'
    cv2.imwrite(output_path2, cv_img2)
    print(f"创建测试图片: {output_path2}")

    return output_path, output_path2

if __name__ == "__main__":
    create_chinese_text_image()
