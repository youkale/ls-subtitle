import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from paddleocr import PaddleOCR
import cv2
import shutil
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher

try:
    from opencc import OpenCC
    HAS_OPENCC = True
except ImportError:
    HAS_OPENCC = False


def get_groups_mean(arr: list, tolerance: float = 20) -> float:
    """
    计算分组后的平均值。
    对给定数组进行分组，每组内的元素与组内最小元素的差值不大于tolerance。
    然后计算最大组的平均值作为结果。

    Args:
        arr: 输入的数字列表
        tolerance: 分组的差值容忍度，默认为20

    Returns:
        最大组的平均值
    """
    if not arr:
        return 0

    arr_sorted = sorted(arr)
    groups = []
    current_group = [arr_sorted[0]]

    for i in range(1, len(arr_sorted)):
        if abs(arr_sorted[i] - current_group[0]) <= tolerance:
            current_group.append(arr_sorted[i])
        else:
            groups.append(current_group)
            current_group = [arr_sorted[i]]

    groups.append(current_group)
    max_group = max(groups, key=len)
    return np.mean(max_group)


def text_similarity(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        相似度分数 (0.0 到 1.0)
    """
    if not text1 or not text2:
        return 0.0

    # 使用 SequenceMatcher 计算相似度
    return SequenceMatcher(None, text1, text2).ratio()


class VideoSubtitleExtractor:
    """视频字幕提取器"""

    def __init__(self, output_dir: str = "output", extract_fps: int = 30,
                 subtitle_region_bottom: float = 0.1, subtitle_region_top: float = 0.45,
                 use_gpu: bool = True, start_time: float = 0, duration: float = None):
        """
        初始化字幕提取器

                Args:
                    output_dir: 输出目录
                    extract_fps: 提取帧率（每秒提取多少帧），默认30
                    subtitle_region_bottom: 字幕区域底部位置（距离底部的百分比），默认0.1（10%）
                    subtitle_region_top: 字幕区域顶部位置（距离底部的百分比），默认0.45（45%）
                    use_gpu: 是否使用GPU加速，默认True
                    start_time: 开始时间（秒），默认0（从头开始）
                    duration: 处理时长（秒），默认None（处理到视频结束）
                """
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.extract_fps = extract_fps
        self.subtitle_region_bottom = subtitle_region_bottom
        self.subtitle_region_top = subtitle_region_top
        self.use_gpu = use_gpu
        self.start_time = start_time
        self.duration = duration

        # 初始化繁体转简体转换器
        if HAS_OPENCC:
            self.cc = OpenCC('t2s')  # 繁体转简体
            print("已启用繁体转简体功能")
        else:
            self.cc = None
            print("未安装opencc，将跳过繁体转简体转换")

        # 检测GPU可用性并设置设备
        if use_gpu:
            gpu_available = self._check_gpu_availability()
            if not gpu_available:
                print("警告: 未检测到可用的GPU，将使用CPU模式")
                self.use_gpu = False
                device = 'cpu'
            else:
                print(f"使用GPU加速进行OCR识别")
                device = 'gpu:0'  # 使用第0块GPU
        else:
            print("使用CPU模式进行OCR识别")
            device = 'cpu'

        # 使用 PP-OCRv5 模型，优化参数以提高识别效果
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # 新版本推荐参数（原use_angle_cls）
            lang='ch',
            text_rec_score_thresh=0.7,      # 识别阈值，平衡敏感度和噪声过滤
            text_det_box_thresh=0.5,        # 检测阈值，适中设置
            text_det_thresh=0.01,            # 像素阈值，适中敏感度
            text_det_unclip_ratio=2.5,      # 扩张系数，扩大文本检测区域
            text_detection_model_name='PP-OCRv5_server_det',
            text_recognition_model_name='PP-OCRv5_server_rec',
            ocr_version='PP-OCRv5',
            device=device  # PaddleOCR 3.2.0+ 使用device参数指定计算设备
        )

    def _check_gpu_availability(self) -> bool:
        """
        检查GPU是否可用

        Returns:
            GPU是否可用
        """
        try:
            import paddle
            return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except Exception as e:
            print(f"GPU检测失败: {e}")
            return False

    def get_video_info(self, video_path: str) -> Dict:
        """
        使用ffprobe获取视频信息

        Args:
            video_path: 视频文件路径

        Returns:
            包含视频信息的字典
        """
        print(f"正在获取视频信息: {video_path}")

        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            '-show_format',
            video_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)

            # 查找视频流
            video_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                raise ValueError("未找到视频流")

            # 解析帧率
            fps_str = video_stream.get('r_frame_rate', '25/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1])

            # 获取总时长
            duration = float(video_info.get('format', {}).get('duration', 0))

            info = {
                'fps': fps,
                'duration': duration,
                'width': video_stream.get('width'),
                'height': video_stream.get('height')
            }

            print(f"视频信息: FPS={info['fps']:.2f}, 时长={info['duration']:.2f}秒, "
                  f"分辨率={info['width']}x{info['height']}")

            return info

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffprobe执行失败: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"解析ffprobe输出失败: {e}")

    def extract_frames(self, video_path: str, fps: float) -> int:
        """
        使用ffmpeg提取视频帧，并裁剪到字幕区域

        Args:
            video_path: 视频文件路径
            fps: 视频帧率

        Returns:
            提取的帧数
        """
        # 构建处理时间信息
        time_info = []
        if self.start_time > 0:
            time_info.append(f"从{self.start_time:.1f}秒开始")
        if self.duration:
            time_info.append(f"处理{self.duration:.1f}秒")
        else:
            time_info.append("处理到视频结束")

        time_desc = "，".join(time_info) if time_info else "处理整个视频"
        print(f"正在提取视频帧（每秒{self.extract_fps}帧），并裁剪到字幕区域...")
        print(f"处理范围: {time_desc}")

        # 获取视频信息
        video_info = self.get_video_info(video_path)
        width = video_info['width']
        height = video_info['height']
        total_duration = video_info['duration']

        # 验证时间参数
        if self.start_time >= total_duration:
            raise ValueError(f"开始时间({self.start_time}s)超过视频总时长({total_duration:.1f}s)")

        # 计算实际处理时长
        actual_duration = self.duration
        if actual_duration:
            if self.start_time + actual_duration > total_duration:
                actual_duration = total_duration - self.start_time
                print(f"注意: 处理时长已调整为{actual_duration:.1f}秒（到视频结束）")

        # 计算裁剪区域（与ocr_frames中的计算保持一致）
        bottom_y = int(height * (1 - self.subtitle_region_top))  # 顶部边界
        top_y = int(height * (1 - self.subtitle_region_bottom))  # 底部边界
        crop_height = top_y - bottom_y

        print(f"视频信息: {width}x{height}, 总时长={total_duration:.1f}秒")
        print(f"字幕区域: y={bottom_y} 到 y={top_y} (高度={crop_height}px, 占比{self.subtitle_region_bottom*100:.0f}%-{self.subtitle_region_top*100:.0f}%)")

        # 创建帧输出目录
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # 使用ffmpeg提取帧，并使用crop filter裁剪到字幕区域
        # crop语法: crop=width:height:x:y
        output_pattern = str(self.frames_dir / "frame_%06d.jpg")

        cmd = [
            'ffmpeg',
        ]

        # 添加开始时间参数（如果指定）
        if self.start_time > 0:
            cmd.extend(['-ss', str(self.start_time)])

        cmd.extend(['-i', video_path])

        # 添加时长参数（如果指定）
        if actual_duration:
            cmd.extend(['-t', str(actual_duration)])

        cmd.extend([
            '-vf', f'fps={self.extract_fps},crop={width}:{crop_height}:0:{bottom_y}',  # 先设置fps再裁剪
            '-q:v', '2',  # 图片质量
            output_pattern,
            '-y'  # 覆盖已存在的文件
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 统计提取的帧数
            frame_count = len(list(self.frames_dir.glob("frame_*.jpg")))
            print(f"成功提取并裁剪 {frame_count} 帧")

            return frame_count

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg执行失败: {e.stderr}")

    def detect_subtitle_region(self, frame_path: str) -> Tuple[int, int, int, int]:
        """
        检测字幕区域（通常在视频底部）

        Args:
            frame_path: 帧图片路径

        Returns:
            字幕区域坐标 (x, y, width, height)
        """
        img = cv2.imread(frame_path)
        height, width = img.shape[:2]

        # 假设字幕在底部20%的区域
        subtitle_height = int(height * 0.2)
        subtitle_y = height - subtitle_height

        return (0, subtitle_y, width, subtitle_height)

    def preview_subtitle_region(self, video_path: str, frame_times: list = None):
        """
        预览字幕区域，生成多个时间点的标注图片

        Args:
            video_path: 视频文件路径
            frame_times: 要预览的时间点列表（秒），如果为None则自动选择多个时间点
        """
        print(f"生成字幕区域预览...")

        # 获取视频时长
        try:
            duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            print(f"视频时长: {duration:.1f}秒")
        except:
            print("警告: 无法获取视频时长，使用默认时间点")
            duration = 60

        # 如果没有指定时间点，自动选择多个时间点
        if frame_times is None:
            # 选择开头、1/4、1/2、3/4、结尾前的时间点
            frame_times = [
                10,  # 开头10秒
                duration * 0.25,  # 1/4处
                duration * 0.5,   # 中间
                duration * 0.75,  # 3/4处
                max(10, duration - 10)  # 结尾前10秒
            ]
            # 去重并排序
            frame_times = sorted(list(set([t for t in frame_times if 0 < t < duration])))

        print(f"将预览 {len(frame_times)} 个时间点:")
        for i, t in enumerate(frame_times, 1):
            print(f"  {i}. {t:.1f}秒")
        print()

        preview_files = []
        crop_files = []

        for idx, time_sec in enumerate(frame_times):
            # 提取指定时间点的帧
            temp_frame = self.frames_dir / f"preview_frame_{idx}.jpg"
            cmd = [
                'ffmpeg',
                '-ss', str(time_sec),
                '-i', video_path,
                '-frames:v', '1',
                str(temp_frame),
                '-y'
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)

                # 读取图片
                img = cv2.imread(str(temp_frame))
                if img is None:
                    print(f"⚠ 无法读取第{idx+1}个预览帧 (时间: {time_sec:.1f}s)")
                    continue

                height, width = img.shape[:2]

                # 计算字幕区域
                bottom_y = int(height * (1 - self.subtitle_region_top))
                top_y = int(height * (1 - self.subtitle_region_bottom))

                # 创建标注图片
                preview_img = img.copy()

                # 绘制字幕区域矩形（绿色）
                cv2.rectangle(preview_img, (0, bottom_y), (width, top_y), (0, 255, 0), 3)

                # 添加文字标注
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(preview_img, f'Subtitle Region (Time: {time_sec:.1f}s)',
                           (10, bottom_y - 10), font, 1, (0, 255, 0), 2)
                cv2.putText(preview_img, f'Bottom: {self.subtitle_region_bottom*100:.0f}%',
                           (10, top_y + 30), font, 0.8, (0, 255, 0), 2)
                cv2.putText(preview_img, f'Top: {self.subtitle_region_top*100:.0f}%',
                           (10, bottom_y - 40), font, 0.8, (0, 255, 0), 2)

                # 裁剪字幕区域
                subtitle_crop = img[bottom_y:top_y, :]

                # 保存完整预览图
                preview_path = self.output_dir / f"preview_region_{idx+1}_{int(time_sec)}s.jpg"
                cv2.imwrite(str(preview_path), preview_img)
                preview_files.append(preview_path)

                # 保存裁剪后的字幕区域
                crop_path = self.output_dir / f"preview_crop_{idx+1}_{int(time_sec)}s.jpg"
                cv2.imwrite(str(crop_path), subtitle_crop)
                crop_files.append(crop_path)

                print(f"✓ 第{idx+1}个预览 (时间: {time_sec:.1f}s) 已生成")

            except subprocess.CalledProcessError as e:
                print(f"✗ 提取第{idx+1}个预览帧失败 (时间: {time_sec:.1f}s): {e.stderr}")

        # 打印总结
        print(f"\n{'='*60}")
        print(f"预览生成完成！共 {len(preview_files)} 个时间点")
        print(f"{'='*60}")
        print(f"\n完整标注图片 ({len(preview_files)}张):")
        for f in preview_files:
            print(f"  - {f}")

        print(f"\n字幕裁剪图片 ({len(crop_files)}张):")
        for f in crop_files:
            print(f"  - {f}")

        if preview_files:
            # 使用第一张图获取尺寸信息
            img = cv2.imread(str(preview_files[0]))
            height, width = img.shape[:2]
            bottom_y = int(height * (1 - self.subtitle_region_top))
            top_y = int(height * (1 - self.subtitle_region_bottom))

            print(f"\n字幕区域信息：")
            print(f"  视频尺寸: {width}x{height}")
            print(f"  字幕区域: y={bottom_y} 到 y={top_y} (高度={top_y-bottom_y}px)")
            print(f"  占比: 底部{self.subtitle_region_bottom*100:.0f}% 到 {self.subtitle_region_top*100:.0f}%")

    def ocr_frames(self) -> Dict[str, Dict]:
        """
        对所有帧进行OCR识别（帧已经是裁剪后的字幕区域）

        Returns:
            识别结果字典，key为帧路径，value包含文本、边界框、置信度等信息
        """
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        results = {}

        # 使用进度条
        for idx, frame_path in enumerate(tqdm(frame_files, desc="OCR Processing", unit="FPS")):
            # 读取图片（已经是裁剪后的字幕区域）
            img = cv2.imread(str(frame_path))
            if img is None:
                tqdm.write(f"警告: 无法读取帧 {frame_path}")
                continue

            # 使用抽象的核心OCR识别方法
            debug_print = (idx == 0) or (frame_path.name == "frame_000708.jpg")  # 第一帧或特定帧打印调试信息
            if debug_print:
                tqdm.write(f"批量识别调试: 处理帧 {frame_path.name}")

            try:
                # 特殊调试：为frame_000708保存调试图片
                if frame_path.name == "frame_000708.jpg":
                    debug_img_path = f"debug_{frame_path.name}"
                    cv2.imwrite(debug_img_path, img)
                    tqdm.write(f"🔍 保存调试图片: {debug_img_path}, 尺寸: {img.shape}")

                ocr_result = self._ocr_image(img, debug_print=debug_print)

                # 从文件名提取真实的帧索引
                frame_name = frame_path.stem  # frame_000708
                real_frame_index = int(frame_name.split('_')[1])  # 708

                # 特殊调试：详细输出frame_000708的OCR结果
                if frame_path.name == "frame_000708.jpg":
                    tqdm.write(f"🔍 frame_000708 OCR原始结果:")
                    tqdm.write(f"  - 识别到的文本数量: {len(ocr_result['texts'])}")
                    for i, text_info in enumerate(ocr_result['texts']):
                        tqdm.write(f"  - 文本{i+1}: \"{text_info['text']}\" -> \"{text_info['simplified_text']}\" (置信度: {text_info['score']:.3f})")
                    tqdm.write(f"  - 合并后文本: \"{ocr_result['combined_text']}\"")

                # 如果有识别结果，保存到字典中
                if ocr_result['texts']:
                    # 按 x 坐标排序（从左到右）
                    text_items = sorted(ocr_result['texts'], key=lambda x: x['box'][0])

                    # 使用已经处理好的合并文本
                    combined_text = ocr_result['combined_text']

                    if debug_print:
                        tqdm.write(f"批量识别调试: 最终文本 '{combined_text}'")

                    # 计算整体边界框
                    if text_items:
                        all_boxes = [item['box'] for item in text_items]
                        xmin = min(box[0] for box in all_boxes)
                        ymin = min(box[1] for box in all_boxes)
                        xmax = max(box[2] for box in all_boxes)
                        ymax = max(box[3] for box in all_boxes)

                        results[str(frame_path)] = {
                            'text': combined_text,
                            'box': [xmin, ymin, xmax, ymax],
                            'frame_index': real_frame_index,  # 使用真实的帧索引
                            'items': text_items  # 保留原始文本项
                        }

            except Exception as e:
                if debug_print:
                    tqdm.write(f"批量识别调试: OCR处理失败 {e}")
                continue

        return results

    def check_ocr_result(self, ocr_result: Dict[str, Dict], video_info: Dict) -> Dict[str, Dict]:
        """
        校验并整合OCR识别结果

        参考: https://github.com/chenwr727/SubErase-Translate-Embed

        主要功能：
        1. 统计字幕的中心位置和高度
        2. 过滤掉不在字幕区域的文本
        3. 合并同一帧内相邻的文本
        4. 填充连续帧之间的空白

        Args:
            ocr_result: OCR识别结果字典
            video_info: 视频信息（包含分辨率）

        Returns:
            校验和整合后的OCR结果
        """
        if not ocr_result:
            return {}

        # 获取图像尺寸（从裁剪后的帧）
        first_frame = next(iter(ocr_result.keys()))
        img = cv2.imread(first_frame)
        if img is None:
            return ocr_result

        img_height, img_width = img.shape[:2]

        # 配置参数 - 大幅放宽容忍度以减少误过滤
        width_delta = img_width * 0.5   # 水平位置容忍度（50%，原30%）
        height_delta = img_height * 0.4  # 垂直位置容忍度（40%，原10%）
        groups_tolerance = img_height * 0.5  # 分组容忍度（50%，原5%）

        x_center_frame = img_width / 2

        # 第一步：统计字幕的中心位置和高度
        center_list = []
        word_height_list = []

        for frame_path, value in tqdm(ocr_result.items(), desc="统计字幕位置", unit="帧"):
            xmin, ymin, xmax, ymax = value['box']
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2

            # 只统计靠近水平中心的文本
            if x_center - width_delta < x_center_frame < x_center + width_delta:
                center_list.append(y_center)
                word_height_list.append(ymax - ymin)

        if not center_list:
            return ocr_result

        # 使用分组统计找到最常见的字幕位置和高度
        center = get_groups_mean(center_list, groups_tolerance)
        word_height = get_groups_mean(word_height_list, groups_tolerance)

        print(f"  检测到字幕中心位置: y={center:.0f}px (容忍±{height_delta:.0f}px)")
        print(f"  检测到字幕平均高度: {word_height:.0f}px")

        # 第二步：过滤并合并同一帧内的文本
        filtered_result = {}

        for frame_path, value in tqdm(ocr_result.items(), desc="过滤OCR结果", unit="帧"):
            xmin, ymin, xmax, ymax = value['box']
            y_center = (ymin + ymax) / 2
            x_center = (xmin + xmax) / 2
            text_height = ymax - ymin

            # 检查是否在字幕区域内
            if (center - height_delta < y_center < center + height_delta and
                word_height - groups_tolerance <= text_height <= word_height + groups_tolerance):

                # 检查多个文本项，看是否需要进一步合并
                if 'items' in value and len(value['items']) > 1:
                    # 合并相邻的文本项
                    merged_text = value['text']
                    merged_box = value['box']
                else:
                    merged_text = value['text']
                    merged_box = value['box']

                filtered_result[frame_path] = {
                    'text': merged_text,
                    'box': merged_box,
                    'frame_index': value['frame_index']
                }

        # 第三步：填充连续帧之间的空白
        if not filtered_result:
            return {}

        # 按帧索引排序
        sorted_frames = sorted(filtered_result.items(), key=lambda x: x[1]['frame_index'])

        final_result = {}
        min_duration_frames = int(self.extract_fps * 0.3)  # 最小持续时间0.3秒

        for i in range(len(sorted_frames)):
            frame_path, value = sorted_frames[i]
            frame_idx = value['frame_index']
            text = value['text']

            final_result[frame_path] = value

            # 如果不是最后一帧，检查是否需要填充
            if i < len(sorted_frames) - 1:
                next_frame_path, next_value = sorted_frames[i + 1]
                next_frame_idx = next_value['frame_index']
                next_text = next_value['text']

                # 如果文本相同且帧间隔不大，填充中间的帧
                if text == next_text and (next_frame_idx - frame_idx) <= min_duration_frames:
                    # 填充中间的帧
                    for fill_idx in range(frame_idx + 1, next_frame_idx):
                        fill_frame_name = f"frame_{fill_idx:06d}.jpg"
                        fill_frame_path = str(self.frames_dir / fill_frame_name)

                        if os.path.exists(fill_frame_path):
                            final_result[fill_frame_path] = {
                                'text': text,
                                'box': value['box'],
                                'frame_index': fill_idx
                            }

        print(f"  过滤前: {len(ocr_result)} 帧，过滤后: {len(filtered_result)} 帧，填充后: {len(final_result)} 帧")

        return final_result

    def merge_subtitle_segments(self, ocr_results: Dict[str, Dict], similarity_threshold: float = 0.8, max_gap_seconds: float = 0.2) -> List[Dict]:
        """
        合并连续相同或相似的字幕段

        改进的合并算法，解决以下问题：
        1. 空文本强制分割问题 - 使用时间间隔判断
        2. 相似度阈值优化 - 降低到0.75以处理更多OCR错误
        3. 时间间隔合并 - 短时间内的相同文本会被合并
        4. 标点符号处理 - 忽略标点符号差异

        Args:
            ocr_results: OCR识别结果字典
            similarity_threshold: 文本相似度阈值（0.0-1.0），默认0.75
            max_gap_seconds: 最大时间间隔（秒），默认0.5秒

        Returns:
            合并后的字幕段列表
        """
        if not ocr_results:
            return []

        print(f"正在合并字幕段（相似度阈值: {similarity_threshold}, 最大间隔: {max_gap_seconds}秒）...")

        # 按帧索引排序
        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        # 第一步：创建初始段落（包含空文本段）
        initial_segments = []
        for frame_path, value in sorted_results:
            text = value['text'].strip()
            frame_idx = value['frame_index']

            initial_segments.append({
                'frame_index': frame_idx,
                'text': text,
                'is_empty': not text
            })

        # 第二步：智能合并算法
        segments = []
        current_segment = None
        text_variants = []

        for i, seg in enumerate(initial_segments):
            text = seg['text']
            frame_idx = seg['frame_index']
            is_empty = seg['is_empty']

            # 如果是空文本，检查是否应该跳过（基于时间间隔）
            if is_empty:
                if current_segment is not None:
                    # 计算时间间隔
                    time_gap = (frame_idx - current_segment['end_frame']) / self.extract_fps

                    # 如果时间间隔很小，跳过这个空文本
                    if time_gap <= max_gap_seconds:
                        # 查看下一个非空文本是否与当前段相似
                        next_text = self._find_next_non_empty_text(initial_segments, i)
                        if next_text and current_segment:
                            current_text = self._normalize_text(current_segment['text'])
                            next_normalized = self._normalize_text(next_text)
                            similarity = text_similarity(current_text, next_normalized)

                            if similarity >= similarity_threshold:
                                # 跳过这个空文本，继续当前段
                                continue

                    # 否则结束当前段
                    self._finalize_current_segment(current_segment, text_variants, segments)
                    current_segment = None
                    text_variants = []
                continue

            # 处理非空文本
            normalized_text = self._normalize_text(text)

            if current_segment is None:
                # 开始新段
                current_segment = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'text': text
                }
                text_variants = [text]
            else:
                # 计算与当前段的相似度
                current_normalized = self._normalize_text(current_segment['text'])
                similarity = text_similarity(normalized_text, current_normalized)

                # 计算时间间隔
                time_gap = (frame_idx - current_segment['end_frame']) / self.extract_fps

                # 判断是否应该合并
                should_merge = (similarity >= similarity_threshold) and (time_gap <= max_gap_seconds)

                if should_merge:
                    # 延长当前段
                    current_segment['end_frame'] = frame_idx
                    text_variants.append(text)
                else:
                    # 结束当前段，开始新段
                    self._finalize_current_segment(current_segment, text_variants, segments)

                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'text': text
                    }
                    text_variants = [text]

        # 保存最后一段
        if current_segment:
            self._finalize_current_segment(current_segment, text_variants, segments)

        print(f"合并后得到 {len(segments)} 个字幕段")
        return segments

    def _normalize_text(self, text: str) -> str:
        """标准化文本，繁体转简体，去除标点符号和空格，用于相似度比较"""
        import re

        # 1. 繁体转简体
        if self.cc and text:
            text = self.cc.convert(text)

        # 2. 去除标点符号和空格
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        return normalized.lower()

    def _convert_to_simplified(self, text: str) -> str:
        """将文本转换为简体中文"""
        if self.cc and text:
            return self.cc.convert(text)
        return text

    def _is_likely_subtitle_by_geometry(self, box_coords: list, text: str, img_height: int, debug_print: bool = False) -> bool:
        """
        基于几何特征判断文本是否可能是字幕

        Args:
            box_coords: 文本边界框坐标 [x1, y1, x2, y2]
            text: 文本内容
            img_height: 图片高度（用于计算相对高度）
            debug_print: 是否打印调试信息

        Returns:
            bool: True表示可能是字幕，False表示可能是噪声
        """
        if len(box_coords) < 4:
            return True  # 如果坐标不完整，默认通过

        x1, y1, x2, y2 = box_coords[:4]
        width = x2 - x1
        height = y2 - y1
        char_count = len(text)

        if width <= 0 or height <= 0 or char_count == 0:
            return True  # 避免除零错误，默认通过

        # 计算几何特征
        aspect_ratio = width / height
        avg_char_width = width / char_count
        relative_height = height / img_height  # 使用实际图片高度计算相对高度

        # 基于分析结果的过滤规则
        is_wide_text = aspect_ratio > 1.6  # 宽高比大于1.6（字幕通常更宽扁）
        is_reasonable_height = relative_height < 0.4  # 相对高度小于40%
        is_reasonable_char_width = avg_char_width < 100  # 平均字符宽度小于100px

        # 特殊规则：过滤明显的车牌模式
        is_license_plate = self._is_license_plate_pattern(text)

        # 单字符特殊处理：对于单个汉字，放宽宽高比要求
        import re
        is_single_char = char_count == 1
        is_chinese_char = bool(re.match(r'^[\u4e00-\u9fff]$', text))

        # 综合判断
        if is_single_char and is_chinese_char:
            # 单个汉字：只检查高度和模式，不检查宽高比
            passes_geometry = is_reasonable_height and is_reasonable_char_width
        else:
            # 多字符：检查所有几何特征
            passes_geometry = is_wide_text and is_reasonable_height and is_reasonable_char_width

        passes_pattern = not is_license_plate

        result = passes_geometry and passes_pattern

        if debug_print:
            print(f"      几何分析: 宽高比={aspect_ratio:.2f}, 相对高度={relative_height:.3f}, 字符宽度={avg_char_width:.1f}")
            print(f"      规则检查: 宽扁={is_wide_text}, 高度合理={is_reasonable_height}, 字符合理={is_reasonable_char_width}")
            print(f"      模式检查: 非车牌={not is_license_plate}")
            print(f"      最终结果: {'通过' if result else '过滤'}")

        return result

    def _is_license_plate_pattern(self, text: str) -> bool:
        """检查是否为车牌号码模式"""
        import re
        # 中国车牌格式：地区码+字母+数字
        license_pattern = r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领海][A-Z]·?\d+$'
        return bool(re.match(license_pattern, text))


    def _ocr_image(self, img, debug_print: bool = False) -> Dict:
        """
        核心的单张图片OCR识别逻辑

        Args:
            img: OpenCV图片对象 (numpy.ndarray)
            debug_print: 是否打印调试信息

        Returns:
            Dict: 包含识别结果的字典
            {
                'texts': [{'text': str, 'simplified_text': str, 'score': float, 'box': list}],
                'combined_text': str,
                'raw_result': OCRResult
            }
        """
        # 获取图片高度用于几何特征计算
        img_height = img.shape[0] if img is not None and hasattr(img, 'shape') else 480

        # 进行OCR识别
        if debug_print:
            print("正在进行OCR识别...")

        try:
            ocr_result = self.ocr.predict(img, use_textline_orientation=True)

            # 解析OCR结果
            result_data = {
                'texts': [],
                'combined_text': "",
                'raw_result': ocr_result
            }

            if ocr_result and len(ocr_result) > 0:
                if debug_print:
                    print(f"OCR识别完成，找到 {len(ocr_result)} 个文本区域")

                for i, item in enumerate(ocr_result):
                    # 尝试不同的方式获取数据
                    rec_texts = []
                    rec_scores = []
                    boxes = []

                    # 方法1: 字典方式
                    if hasattr(item, 'get'):
                        rec_texts = item.get('rec_texts', [])
                        rec_scores = item.get('rec_scores', [])

                        # 优先使用现成的 rec_boxes
                        rec_boxes = item.get('rec_boxes')
                        if rec_boxes is not None and hasattr(rec_boxes, 'shape'):
                            boxes = rec_boxes.tolist()
                        else:
                            # 备选方案：从 dt_polys 计算边界框
                            dt_polys = item.get('dt_polys', [])
                            if dt_polys:
                                for poly in dt_polys:
                                    x_coords = [p[0] for p in poly]
                                    y_coords = [p[1] for p in poly]
                                    xmin = int(min(x_coords))
                                    xmax = int(max(x_coords))
                                    ymin = int(min(y_coords))
                                    ymax = int(max(y_coords))
                                    boxes.append([xmin, ymin, xmax, ymax])
                            else:
                                # 最后尝试获取boxes字段
                                boxes = item.get('boxes', [])

                    # 方法2: 属性方式
                    if not rec_texts:
                        rec_texts = getattr(item, 'rec_texts', [])
                        rec_scores = getattr(item, 'rec_scores', [])
                        dt_polys = getattr(item, 'dt_polys', [])

                        # 如果有检测框，转换为边界框格式
                        if dt_polys:
                            for poly in dt_polys:
                                x_coords = [p[0] for p in poly]
                                y_coords = [p[1] for p in poly]
                                xmin = int(min(x_coords))
                                xmax = int(max(x_coords))
                                ymin = int(min(y_coords))
                                ymax = int(max(y_coords))
                                boxes.append([xmin, ymin, xmax, ymax])
                        else:
                            boxes = getattr(item, 'boxes', [])

                    # 方法3: 尝试其他可能的属性名
                    if not rec_texts:
                        # 尝试直接访问文本内容
                        if hasattr(item, 'text'):
                            rec_texts = [item.text] if item.text else []
                            rec_scores = [getattr(item, 'score', 1.0)] if item.text else []
                        elif hasattr(item, 'texts'):
                            rec_texts = item.texts
                            rec_scores = getattr(item, 'scores', [1.0] * len(rec_texts))

                    # 如果boxes为空，创建默认的boxes
                    if rec_texts and (not boxes or len(boxes) != len(rec_texts)):
                        boxes = [[0, 0, 100, 30] for _ in rec_texts]

                    if rec_texts and rec_scores:
                        for j, (text, score, box) in enumerate(zip(rec_texts, rec_scores, boxes)):
                            if debug_print:
                                print(f"  检测到文本 {j+1}: \"{text}\" (置信度: {score:.3f})")

                            if score > 0.5:  # 置信度阈值
                                # 几何特征过滤
                                box_coords = box if isinstance(box, list) else box.tolist() if hasattr(box, 'tolist') else [0, 0, 100, 30]
                                if self._is_likely_subtitle_by_geometry(box_coords, text, img_height, debug_print):
                                    # 转换为简体中文
                                    simplified_text = self._convert_to_simplified(text)

                                    text_info = {
                                        'text': text,
                                        'simplified_text': simplified_text,
                                        'score': float(score),
                                        'box': box_coords
                                    }
                                    result_data['texts'].append(text_info)

                                    if debug_print:
                                        print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f})")
                                elif debug_print:
                                    print(f"    ❌ 几何过滤: \"{text}\" (置信度: {score:.3f})")
                            else:
                                if debug_print:
                                    print(f"    ✗ 跳过: 置信度过低 ({score:.3f} < 0.5)")

                # 合并所有文本
                if result_data['texts']:
                    # 先合并文本，然后去除所有空白符
                    combined_text = ''.join([item['simplified_text'] for item in result_data['texts']])
                    # 去除所有空白符（空格、换行、制表符等）
                    combined_text = ''.join(combined_text.split())
                    result_data['combined_text'] = combined_text

                    if debug_print:
                        print(f"\n合并文本: \"{combined_text}\"")
                else:
                    if debug_print:
                        print("\n未识别到任何文本")
            else:
                if debug_print:
                    print("OCR识别完成，但未找到任何文本")

            return result_data

        except Exception as e:
            if debug_print:
                print(f"OCR识别失败: {e}")
            raise

    def _find_next_non_empty_text(self, segments: List[Dict], start_index: int) -> str:
        """查找下一个非空文本"""
        for i in range(start_index + 1, len(segments)):
            if not segments[i]['is_empty']:
                return segments[i]['text']
        return ""

    def _is_likely_same_text(self, text1: str, text2: str) -> bool:
        """判断两个文本是否可能是同一句话（考虑OCR常见错误）"""
        if not text1 or not text2:
            return False

        # 长度差异太大，不太可能是同一句话
        if abs(len(text1) - len(text2)) > max(len(text1), len(text2)) * 0.3:
            return False

        # 检查是否有足够的共同字符
        common_chars = set(text1) & set(text2)
        min_len = min(len(text1), len(text2))

        return len(common_chars) >= min_len * 0.6


    def _finalize_current_segment(self, current_segment: Dict, text_variants: List[str], segments: List[Dict]):
        """完成当前段落并添加到结果中"""
        if not current_segment:
            return

        # 选择最佳文本（出现次数最多，或最长的）
        if text_variants:
            # 优先选择出现次数最多的
            text_counts = {}
            for text in text_variants:
                normalized = self._normalize_text(text)
                if normalized not in text_counts:
                    text_counts[normalized] = []
                text_counts[normalized].append(text)

            # 选择出现次数最多的组，然后选择该组中最长的文本
            best_group = max(text_counts.values(), key=len)
            final_text = max(best_group, key=len)
        else:
            final_text = current_segment['text']

        # 计算时间戳
        start_time = current_segment['start_frame'] / self.extract_fps + self.start_time
        end_time = (current_segment['end_frame'] + 1) / self.extract_fps + self.start_time

        segments.append({
            'text': final_text,
            'start_time': start_time,
            'end_time': end_time
        })

    def generate_raw_segments(self, ocr_results: Dict[str, Dict]) -> List[Dict]:
        """
        生成未合并的原始字幕段（每帧一个段落）
        用于调试和对比

        Args:
            ocr_results: OCR识别结果字典

        Returns:
            原始字幕段列表
        """
        if not ocr_results:
            return []

        segments = []
        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        for frame_path, value in sorted_results:
            text = value['text'].strip()
            frame_idx = value['frame_index']

            if not text:
                continue

            # 每帧作为独立的段落
            start_time = frame_idx / self.extract_fps + self.start_time
            end_time = (frame_idx + 1) / self.extract_fps + self.start_time

            segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time
            })

        return segments

    def format_timestamp(self, seconds: float) -> str:
        """
        将秒数转换为SRT时间格式

        Args:
            seconds: 秒数

        Returns:
            SRT格式时间字符串 (HH:MM:SS,mmm)
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def generate_srt(self, segments: List[Dict], output_path: str):
        """
        生成SRT字幕文件

        Args:
            segments: 字幕段列表
            output_path: 输出文件路径
        """
        print(f"正在生成SRT文件: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, segment in enumerate(segments, 1):
                start_time = self.format_timestamp(segment['start_time'])
                end_time = self.format_timestamp(segment['end_time'])
                text = segment['text']

                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n")
                f.write("\n")

        print(f"SRT文件已保存")

    def process_video(self, video_path: str, output_srt_path: str = None, debug_raw: bool = False):
        """
        处理视频并生成SRT字幕

        Args:
            video_path: 视频文件路径
            output_srt_path: 输出SRT文件路径（可选）
            debug_raw: 是否输出未合并的原始OCR结果（用于调试）
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确定输出文件路径并转换为 Path 对象
        if output_srt_path is None:
            output_srt_path = self.output_dir / f"{video_path.stem}.srt"
        else:
            output_srt_path = Path(output_srt_path)

        try:
            # 步骤1: 获取视频信息
            video_info = self.get_video_info(str(video_path))

            # 步骤2: 提取视频帧
            frame_count = self.extract_frames(str(video_path), video_info['fps'])

            if frame_count == 0:
                raise RuntimeError("未能提取任何视频帧")

            # 步骤3: OCR识别
            ocr_results = self.ocr_frames()

            # 步骤4: 检查和优化OCR结果
            print("正在检查OCR结果...")
            ocr_results = self.check_ocr_result(ocr_results, video_info)

            # 步骤4.5: 如果需要，输出原始未合并的调试文件
            if debug_raw:
                raw_output_path = output_srt_path.parent / f"{output_srt_path.stem}_raw.srt"
                print(f"\n生成原始OCR调试文件: {raw_output_path}")
                raw_segments = self.generate_raw_segments(ocr_results)
                self.generate_srt(raw_segments, str(raw_output_path))
                print(f"原始段落数: {len(raw_segments)} 个")

            # 步骤5: 合并字幕段
            segments = self.merge_subtitle_segments(ocr_results)

            # 步骤6: 生成SRT文件
            self.generate_srt(segments, str(output_srt_path))

            print(f"\n处理完成！")
            print(f"SRT文件: {output_srt_path}")

            # 清理临时帧文件（可选）
            # shutil.rmtree(self.frames_dir)

        except Exception as e:
            print(f"处理失败: {e}")
            raise

    def ocr_single_image(self, image_path: str, crop_region: bool = False, save_result: bool = False) -> Dict:
        """
        对单张图片进行OCR识别

        Args:
            image_path: 图片文件路径
            crop_region: 是否裁剪到字幕区域
            save_result: 是否保存结果到文件

        Returns:
            OCR识别结果
        """
        import cv2
        from pathlib import Path

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        print(f"正在识别图片: {image_path}")

        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片文件: {image_path}")

        height, width = img.shape[:2]
        print(f"图片尺寸: {width}x{height}")

        # 如果需要裁剪到字幕区域
        if crop_region:
            # 计算字幕区域
            bottom_y = int(height * (1 - self.subtitle_region_bottom))
            top_y = int(height * (1 - self.subtitle_region_top))
            crop_height = bottom_y - top_y

            print(f"字幕区域: y={top_y} 到 y={bottom_y} (高度={crop_height}px, 占比{self.subtitle_region_bottom*100:.1f}%-{self.subtitle_region_top*100:.1f}%)")

            # 裁剪图片
            cropped_img = img[top_y:bottom_y, 0:width]

            # 保存裁剪后的图片用于调试
            crop_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"
            cv2.imwrite(str(crop_path), cropped_img)
            print(f"裁剪后的图片已保存到: {crop_path}")

            # 使用裁剪后的图片进行OCR
            ocr_img = cropped_img
        else:
            print("使用完整图片进行OCR识别")
            ocr_img = img

        # 使用抽象的核心OCR识别方法
        try:
            core_result = self._ocr_image(ocr_img, debug_print=True)

            # 构建完整的结果数据
            result_data = {
                'image_path': str(image_path),
                'image_size': f"{width}x{height}",
                'cropped': crop_region,
                'texts': core_result['texts'],
                'combined_text': core_result['combined_text'],
                'raw_result': core_result['raw_result']
            }

            # 保存结果到文件
            if save_result:
                result_file = image_path.parent / f"{image_path.stem}_ocr_result.json"

                # 创建可序列化的结果
                save_data = {
                    'image_path': result_data['image_path'],
                    'image_size': result_data['image_size'],
                    'cropped': result_data['cropped'],
                    'combined_text': result_data['combined_text'],
                    'texts': result_data['texts'],
                    'text_count': len(result_data['texts'])
                }

                import json
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

                print(f"识别结果已保存到: {result_file}")

                # 同时保存简单的文本文件
                text_file = image_path.parent / f"{image_path.stem}_ocr_result.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"图片: {image_path}\n")
                    f.write(f"尺寸: {result_data['image_size']}\n")
                    f.write(f"裁剪: {'是' if crop_region else '否'}\n")
                    f.write(f"识别文本数: {len(result_data['texts'])}\n\n")
                    f.write(f"合并文本: {result_data['combined_text']}\n\n")
                    f.write("详细结果:\n")
                    for i, text_info in enumerate(result_data['texts'], 1):
                        f.write(f"{i}. \"{text_info['simplified_text']}\" (置信度: {text_info['score']:.3f})\n")

                print(f"文本结果已保存到: {text_file}")

            return result_data

        except Exception as e:
            print(f"OCR识别失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用PaddleOCR从视频中提取字幕并生成SRT文件'
    )
    # 创建互斥组：视频处理 vs 单图OCR
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        'video',
        nargs='?',
        type=str,
        help='输入视频文件路径'
    )
    group.add_argument(
        '--ocr-image',
        type=str,
        help='单张图片OCR识别模式：指定图片文件路径'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出SRT文件路径（默认: output/<视频名>.srt）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='输出目录（默认: output）'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='提取帧率，每秒提取多少帧（默认: 30）'
    )
    parser.add_argument(
        '--subtitle-bottom',
        type=float,
        default=0.2,
        help='字幕区域底部位置，距离底部的百分比（默认: 0.1，即10%%）'
    )
    parser.add_argument(
        '--subtitle-top',
        type=float,
        default=0.45,
        help='字幕区域顶部位置，距离底部的百分比（默认: 0.45，即45%%）'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        default=True,
        help='使用GPU加速（默认开启，需要安装paddlepaddle-gpu）'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='强制使用CPU模式'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='仅预览字幕区域，不进行OCR识别（用于调试字幕位置）'
    )
    parser.add_argument(
        '--preview-times',
        type=str,
        default=None,
        help='预览模式下的时间点（秒），用逗号分隔，如 "10,30,60,90"。不指定则自动选择多个时间点'
    )
    parser.add_argument(
        '--start-time',
        type=float,
        default=0,
        help='开始处理的时间点（秒），默认: 0（从头开始）'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='处理的时长（秒），默认: None（处理到视频结束）'
    )
    parser.add_argument(
        '--debug-raw',
        action='store_true',
        help='输出未合并的原始OCR结果到 *_raw.srt 文件（用于调试）'
    )
    parser.add_argument(
        '--crop-region',
        action='store_true',
        help='单图OCR模式：是否裁剪到字幕区域（默认处理整张图片）'
    )
    parser.add_argument(
        '--save-result',
        action='store_true',
        help='单图OCR模式：保存识别结果到文本文件'
    )

    args = parser.parse_args()

# 创建提取器（用于OCR功能）
    extractor = VideoSubtitleExtractor(
        output_dir=args.output_dir,
        extract_fps=args.fps,
        subtitle_region_bottom=args.subtitle_bottom,
        subtitle_region_top=args.subtitle_top,
        use_gpu=not args.cpu,
        start_time=0,
        duration=None
    )

    # 检查是否是单图OCR模式
    if args.ocr_image:
        print("=" * 50)
        print("单张图片OCR识别模式")
        print("=" * 50)

        try:
            # 执行单图OCR
            result = extractor.ocr_single_image(
                image_path=args.ocr_image,
                crop_region=args.crop_region,
                save_result=args.save_result
            )

            print("\n" + "=" * 50)
            print("OCR识别完成")
            print("=" * 50)

            if result['combined_text']:
                print(f"✓ 识别成功: \"{result['combined_text']}\"")
            else:
                print("✗ 未识别到任何文本")

        except Exception as e:
            print(f"✗ OCR识别失败: {e}")
            return 1

        return 0

    # 验证视频模式的参数
    if not args.video:
        parser.error("视频模式需要提供视频文件路径")

    # 验证字幕区域参数
    if args.subtitle_bottom < 0 or args.subtitle_bottom > 1:
        parser.error("--subtitle-bottom 必须在 0 和 1 之间")
    if args.subtitle_top < 0 or args.subtitle_top > 1:
        parser.error("--subtitle-top 必须在 0 和 1 之间")
    if args.subtitle_bottom >= args.subtitle_top:
        parser.error("--subtitle-bottom 必须小于 --subtitle-top")

    # 验证时间参数
    if args.start_time < 0:
        parser.error("--start-time 必须 >= 0")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration 必须 > 0")

    # 如果是预览模式
    if args.preview:
        print("=" * 50)
        print("字幕区域预览模式")
        print("=" * 50)
        extractor.output_dir.mkdir(parents=True, exist_ok=True)
        extractor.frames_dir.mkdir(parents=True, exist_ok=True)

        # 解析预览时间点
        preview_times = None
        if args.preview_times:
            try:
                preview_times = [float(t.strip()) for t in args.preview_times.split(',')]
                print(f"使用指定的时间点: {preview_times}")
            except ValueError:
                print(f"警告: 无法解析时间点 '{args.preview_times}'，将自动选择时间点")

        extractor.preview_subtitle_region(args.video, preview_times)
        print("\n提示：检查生成的图片，如果字幕位置不对，请调整 --subtitle-bottom 和 --subtitle-top 参数")
        return

    # 处理视频
    extractor.process_video(args.video, args.output, debug_raw=args.debug_raw)


if __name__ == "__main__":
    main()
