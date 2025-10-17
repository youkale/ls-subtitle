#!/usr/bin/env python3
"""批量检查examples/1/目录下所有图片的OCR识别结果"""

import os
import sys
from pathlib import Path
from main import VideoSubtitleExtractor

def main():
    # 初始化OCR提取器
    print("正在初始化OCR引擎...")
    extractor = VideoSubtitleExtractor(
        output_dir="output",
        extract_fps=15,
        use_gpu=True
    )

    # 获取所有原始图片
    examples_dir = Path("examples/1")
    image_files = sorted([
        f for f in examples_dir.glob("frame_*.jpg")
        if "_ocr_result" not in f.name and "_visualization" not in f.name
    ])

    print(f"\n{'='*80}")
    print(f"批量OCR识别 - examples/1/ 目录")
    print(f"{'='*80}")
    print(f"共找到 {len(image_files)} 张图片\n")

    results = []

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")
        print("-" * 80)

        try:
            # 读取图片
            import cv2
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"❌ 无法读取图片")
                results.append({
                    'file': image_path.name,
                    'status': 'error',
                    'text': 'Unable to read image'
                })
                continue

            # 进行OCR识别（不打印调试信息）
            ocr_result = extractor._ocr_image(
                img,
                debug_print=False,
                apply_x_filter=True,
                frame_path=str(image_path)
            )

            combined_text = ocr_result['combined_text']

            if combined_text:
                print(f"✓ 识别成功: 「{combined_text}」")
                results.append({
                    'file': image_path.name,
                    'status': 'success',
                    'text': combined_text
                })
            else:
                print(f"○ 未识别到文本")
                results.append({
                    'file': image_path.name,
                    'status': 'empty',
                    'text': ''
                })

        except Exception as e:
            print(f"❌ 识别失败: {e}")
            results.append({
                'file': image_path.name,
                'status': 'error',
                'text': str(e)
            })

    # 打印汇总
    print(f"\n\n{'='*80}")
    print(f"识别结果汇总")
    print(f"{'='*80}\n")

    success_count = sum(1 for r in results if r['status'] == 'success')
    empty_count = sum(1 for r in results if r['status'] == 'empty')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"总计: {len(results)} 张图片")
    print(f"  ✓ 成功识别: {success_count} 张")
    print(f"  ○ 无文本: {empty_count} 张")
    print(f"  ❌ 识别失败: {error_count} 张")

    # 打印所有成功识别的文本
    print(f"\n{'='*80}")
    print(f"所有识别出的文本")
    print(f"{'='*80}\n")

    for result in results:
        if result['status'] == 'success' and result['text']:
            print(f"{result['file']:30s} → {result['text']}")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
