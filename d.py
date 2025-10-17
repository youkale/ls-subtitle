from main import VideoSubtitleExtractor
import cv2
from pathlib import Path

extractor = VideoSubtitleExtractor(output_dir='output', extract_fps=15, use_gpu=True)

# 识别所有examples目录
examples_base = Path('examples')
all_results = {}

for example_dir in sorted([d for d in examples_base.iterdir() if d.is_dir()]):
    dir_name = example_dir.name
    image_files = sorted([f for f in example_dir.glob('frame_*.jpg')
                         if '_ocr_result' not in f.name and '_visualization' not in f.name])

    if not image_files:
        continue

    print(f'\n处理 examples/{dir_name}/ (共{len(image_files)}张)')

    results = {}
    for idx, img_path in enumerate(image_files, 1):
        print(f'  [{idx}/{len(image_files)}] {img_path}')
        img = cv2.imread(str(img_path))
        if img is not None:
            result = extractor._ocr_image(img, debug_print=False, apply_x_filter=True, frame_path=str(img_path))
            text = result['combined_text']
            results[img_path.name] = text
            status = '✓' if text else '○'
            print(f'       {status} 识别结果: {text if text else "(无文本)"}')

    all_results[dir_name] = results
    success = sum(1 for t in results.values() if t)
    print(f'  识别成功: {success}/{len(results)} 张')

# 打印汇总报告
print('\n\n' + '='*80)
print('完整识别报告')
print('='*80 + '\n')

total_images = 0
total_success = 0

for dir_name, results in sorted(all_results.items()):
    print(f'examples/{dir_name}/')
    print('-'*80)

    success = sum(1 for t in results.values() if t)
    total_images += len(results)
    total_success += success

    print(f'  总计: {len(results)} 张图片')
    print(f'  识别到文本: {success} 张')
    print(f'  识别率: {success/len(results)*100:.1f}%')

    # 统计唯一文本
    unique_texts = set(t for t in results.values() if t)
    if unique_texts:
        print(f'  识别到的文本:')
        for text in sorted(unique_texts):
            count = sum(1 for t in results.values() if t == text)
            print(f'    - 「{text}」 ({count}次)')
    else:
        print(f'  无识别文本')
    print()

print('='*80)
print(f'总计: {total_images} 张图片，成功识别 {total_success} 张 ({total_success/total_images*100:.1f}%)')
print('='*80)
