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


def smart_chinese_similarity(text1: str, text2: str) -> float:
    """
    智能中文相似度计算：
    - 对于2-4个汉字，只要有(长度-1)个字符正确就认为匹配成功
    - 其他情况使用标准相似度计算

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        相似度分数 (0.0 到 1.0)
    """
    if not text1 or not text2:
        return 0.0

    # 去除空格和标点，只保留中文字符
    import re
    clean1 = re.sub(r'[^\u4e00-\u9fff]', '', text1)
    clean2 = re.sub(r'[^\u4e00-\u9fff]', '', text2)

    if not clean1 or not clean2:
        return text_similarity(text1, text2)

    # 对于2-4个汉字的特殊处理
    if 2 <= len(clean1) <= 4 and 2 <= len(clean2) <= 4:
        # 如果长度相同，计算匹配字符数
        if len(clean1) == len(clean2):
            matches = sum(1 for c1, c2 in zip(clean1, clean2) if c1 == c2)
            required_matches = len(clean1) - 1  # 需要(长度-1)个字符匹配

            if matches >= required_matches:
                return 1.0  # 认为完全匹配
            else:
                return matches / len(clean1)  # 返回实际匹配比例

        # 长度不同时，使用编辑距离的思想
        else:
            # 计算最长公共子序列
            def lcs_length(s1, s2):
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]

                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

                return dp[m][n]

            lcs_len = lcs_length(clean1, clean2)
            min_len = min(len(clean1), len(clean2))

            # 如果公共字符数 >= min_len - 1，认为匹配
            if lcs_len >= min_len - 1:
                return 1.0
            else:
                return lcs_len / max(len(clean1), len(clean2))

    # 其他情况使用标准相似度
    return text_similarity(text1, text2)


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
            text_rec_score_thresh=0.8,      # 识别阈值，平衡敏感度和噪声过滤
            text_det_box_thresh=0.5,        # 检测阈值，适中设置
            text_det_thresh=0.1,            # 像素阈值，适中敏感度
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
            debug_print = (idx == 0)  # 第一帧打印调试信息
            if debug_print:
                tqdm.write(f"批量识别调试: 处理帧 {frame_path.name}")

            try:
                ocr_result = self._ocr_image(img, debug_print=debug_print)

                # 从文件名提取真实的帧索引
                frame_name = frame_path.stem  # frame_000708
                real_frame_index = int(frame_name.split('_')[1])  # 708

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
                            'items': text_items,  # 保留原始文本项
                            'raw_result': ocr_result['raw_result']  # 保存原始OCR结果用于新合并算法
                        }
                else:
                    # 即使没有识别出文字，也要保存帧信息和原始结果（用于间隙填充）
                    results[str(frame_path)] = {
                        'text': '',
                        'box': [],
                        'frame_index': real_frame_index,
                        'items': [],
                        'raw_result': ocr_result['raw_result']  # 保存原始OCR结果
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

    def merge_subtitle_segments(self, ocr_results: Dict[str, Dict], similarity_threshold: float = 0.8, gap_time_threshold: float = 2.0) -> List[Dict]:
        """
        智能合并字幕段，集成过滤与合并逻辑

        新的合并算法特性：
        1. 智能相似度：2-4个汉字只要有(长度-1)个字符正确就匹配
        2. 间隙填充：OCR识别失败时，如果有检测区域但无识别文字，延续前一帧文字
        3. X轴中心点过滤：只处理通过图片X轴中心的文本区域
        4. 繁体转简体：统一转换为简体中文进行比较

        Args:
            ocr_results: OCR识别结果字典，包含所有帧的识别信息
            similarity_threshold: 文本相似度阈值（0.0-1.0），默认0.8
            gap_time_threshold: 间隙填充的最大时间阈值（秒），默认2.0秒

        Returns:
            合并后的字幕段列表
        """
        if not ocr_results:
            return []

        print(f"正在执行智能合并算法（相似度阈值: {similarity_threshold}，间隙填充阈值: {gap_time_threshold}s）...")

        # 按帧索引排序，获取所有帧信息
        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        # 第一步：处理所有帧，包括空帧和有检测区域但无识别文字的帧
        processed_frames = []
        for frame_path, value in sorted_results:
            frame_idx = value['frame_index']
            text = value['text'].strip() if 'text' in value else ""

            # 检查是否有检测区域（即使没有识别出文字）
            has_detection_region = False
            is_valid_subtitle_region = False

            if 'raw_result' in value and value['raw_result']:
                # 检查是否有检测到的文本区域
                raw_result = value['raw_result']
                if hasattr(raw_result, 'dt_polys') and raw_result.dt_polys:
                    has_detection_region = True
                    # 检查检测区域是否通过X轴中心点（过滤逻辑）
                    is_valid_subtitle_region = self._check_detection_regions_validity(raw_result.dt_polys, frame_path)

            # 转换为简体中文
            simplified_text = ""
            if text:
                simplified_text = self._convert_to_simplified(text)

            processed_frames.append({
                'frame_index': frame_idx,
                'frame_path': frame_path,
                'original_text': text,
                'simplified_text': simplified_text,
                'has_detection_region': has_detection_region,
                'is_valid_subtitle_region': is_valid_subtitle_region,
                'has_recognized_text': bool(simplified_text)
            })

        if not processed_frames:
            return []

        # 第二步：间隙填充算法
        filled_frames = self._fill_recognition_gaps(processed_frames, gap_time_threshold)

        # 第三步：智能合并算法
        segments = []
        current_segment = None
        text_variants = []

        for frame_data in filled_frames:
            frame_idx = frame_data['frame_index']
            text = frame_data['final_text']

            # 跳过无效的字幕区域或空文本
            if not text or not frame_data.get('is_valid_subtitle_region', True):
                continue

            if current_segment is None:
                # 开始新段
                current_segment = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'text': text
                }
                text_variants = [text]
            else:
                # 计算时间戳
                current_end_time = (current_segment['end_frame'] + 1) / self.extract_fps + self.start_time
                new_start_time = frame_idx / self.extract_fps + self.start_time

                # 使用智能相似度计算
                similarity = smart_chinese_similarity(current_segment['text'], text)

                # 计算时间间隔
                time_gap = new_start_time - current_end_time

                # 判断是否应该合并的条件：
                # 1. 严格时间连续：前一句结束时间 = 当前句开始时间 且 文本相似
                # 2. 短时间间隔：time_gap < 150ms 且 文本相似（处理1帧识别失败的情况）
                is_time_continuous = abs(time_gap) < 0.001  # 允许1毫秒的浮点误差
                is_short_gap = 0 < time_gap < 0.15  # 150毫秒内的短间隔
                is_similar = similarity >= similarity_threshold

                should_merge = is_similar and (is_time_continuous or is_short_gap)

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

        print(f"智能合并完成，得到 {len(segments)} 个字幕段")
        return segments

    def _check_detection_regions_validity(self, dt_polys, frame_path: str) -> bool:
        """
        检查检测区域是否为有效的字幕区域（基于X轴中心点过滤）

        Args:
            dt_polys: PaddleOCR检测到的多边形区域列表
            frame_path: 帧文件路径（用于获取图片尺寸）

        Returns:
            bool: 是否包含有效的字幕区域
        """
        if not dt_polys:
            return False

        try:
            # 尝试从帧路径获取图片尺寸，或使用默认值
            img_width = 1080  # 默认宽度

            # 如果可以读取图片，获取实际宽度
            try:
                import cv2
                img = cv2.imread(frame_path)
                if img is not None:
                    img_width = img.shape[1]
            except:
                pass  # 使用默认宽度

            center_x = img_width / 2
            tolerance_ratio = 0.15  # 15%的容忍度

            for poly in dt_polys:
                # 将多边形转换为边界框
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                elif not isinstance(poly, list):
                    continue

                # 获取边界框的左右边界
                x_coords = [point[0] for point in poly if len(point) >= 2]
                if not x_coords:
                    continue

                left_x = min(x_coords)
                right_x = max(x_coords)

                # 检查X轴中心点是否穿过这个区域
                if left_x <= center_x <= right_x:
                    # 检查左右平衡性
                    left_width = center_x - left_x
                    right_width = right_x - center_x
                    total_width = right_x - left_x

                    if total_width > 0:
                        left_ratio = left_width / total_width
                        right_ratio = right_width / total_width

                        # 左右比例都应该在合理范围内（15%-85%）
                        if (tolerance_ratio <= left_ratio <= (1 - tolerance_ratio) and
                            tolerance_ratio <= right_ratio <= (1 - tolerance_ratio)):
                            return True

            return False

        except Exception as e:
            # 出错时保守处理，认为是有效区域
            return True

    def _fill_recognition_gaps(self, processed_frames: List[Dict], gap_time_threshold: float) -> List[Dict]:
        """
        填充OCR识别失败的间隙

        当检测到文本区域但没有识别出文字时，如果前一帧有识别结果，
        且时间间隔在阈值内，则延续前一帧的文字。

        Args:
            processed_frames: 处理过的帧数据列表
            gap_time_threshold: 间隙填充的最大时间阈值（秒）

        Returns:
            填充后的帧数据列表
        """
        if not processed_frames:
            return []

        filled_frames = []
        last_valid_text = ""
        last_valid_frame_idx = -1

        for frame_data in processed_frames:
            frame_idx = frame_data['frame_index']
            has_text = frame_data['has_recognized_text']
            has_detection = frame_data['has_detection_region']
            is_valid_region = frame_data['is_valid_subtitle_region']
            simplified_text = frame_data['simplified_text']

            # 计算当前帧的时间
            current_time = frame_idx / self.extract_fps + self.start_time
            last_valid_time = last_valid_frame_idx / self.extract_fps + self.start_time if last_valid_frame_idx >= 0 else 0

            final_text = simplified_text
            is_gap_filled = False

            # 间隙填充逻辑
            if (not has_text and has_detection and is_valid_region and
                last_valid_text and last_valid_frame_idx >= 0):

                time_gap = current_time - last_valid_time

                # 如果时间间隔在阈值内，延续前一帧的文字
                if time_gap <= gap_time_threshold:
                    final_text = last_valid_text
                    is_gap_filled = True

            # 更新最后有效的文字和帧索引
            if has_text and simplified_text:
                last_valid_text = simplified_text
                last_valid_frame_idx = frame_idx

            # 创建填充后的帧数据
            filled_frame_data = frame_data.copy()
            filled_frame_data['final_text'] = final_text
            filled_frame_data['is_gap_filled'] = is_gap_filled

            filled_frames.append(filled_frame_data)

        # 统计填充信息
        gap_filled_count = sum(1 for f in filled_frames if f.get('is_gap_filled', False))
        if gap_filled_count > 0:
            print(f"  间隙填充: 填充了 {gap_filled_count} 个识别失败的帧")

        return filled_frames

    def _normalize_text(self, text: str) -> str:
        """标准化文本，繁体转简体，去除标点符号和空格，用于相似度比较"""
        import re

        # 1. 繁体转简体
        if self.cc and text:
            text = self.cc.convert(text)

        # 2. 去除标点符号和空格
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', text)
        return normalized.lower()

    def _serialize_ocr_result(self, ocr_result) -> dict:
        """
        将PaddleOCR的原始结果转换为可序列化的格式

        Args:
            ocr_result: PaddleOCR的原始结果

        Returns:
            可序列化的字典格式结果
        """
        if not ocr_result or len(ocr_result) == 0:
            return {
                'status': 'no_result',
                'message': '未检测到任何文本'
            }

        try:
            # 处理OCR结果（可能是list格式）
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                item = ocr_result[0]

                serialized_result = {
                    'status': 'success',
                    'detection_count': len(item.get('dt_polys', [])) if 'dt_polys' in item else 0,
                    'recognition_count': len(item.get('rec_texts', [])) if 'rec_texts' in item else 0,
                    'dt_polys': [],
                    'rec_texts': item.get('rec_texts', []),
                    'rec_scores': item.get('rec_scores', []),
                    'rec_boxes': [],
                    'textline_orientation_angles': item.get('textline_orientation_angles', [])
                }

                # 转换检测框坐标
                if 'dt_polys' in item and item['dt_polys']:
                    for poly in item['dt_polys']:
                        if hasattr(poly, 'tolist'):
                            serialized_result['dt_polys'].append(poly.tolist())
                        else:
                            serialized_result['dt_polys'].append(poly)

                # 转换识别框坐标
                if 'rec_boxes' in item and item['rec_boxes'] is not None:
                    if hasattr(item['rec_boxes'], 'tolist'):
                        serialized_result['rec_boxes'] = item['rec_boxes'].tolist()
                    else:
                        serialized_result['rec_boxes'] = item['rec_boxes']

                # 转换分数为普通float
                if 'rec_scores' in item and item['rec_scores']:
                    serialized_result['rec_scores'] = [float(score) for score in item['rec_scores']]

                return serialized_result
            else:
                return {
                    'status': 'unknown_format',
                    'message': '未知的OCR结果格式',
                    'raw_type': str(type(ocr_result))
                }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'序列化OCR结果时出错: {str(e)}',
                'error_type': type(e).__name__
            }

    def _convert_to_simplified(self, text: str) -> str:
        """将文本转换为简体中文"""
        if self.cc and text:
            return self.cc.convert(text)
        return text

    def visualize_ocr_result(self, json_file_path: str) -> str:
        """
        根据OCR结果JSON文件在原图上绘制检测和识别区域

        Args:
            json_file_path: OCR结果JSON文件路径

        Returns:
            可视化结果图片的保存路径
        """
        import json
        import cv2
        import numpy as np
        from pathlib import Path

        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON文件不存在: {json_path}")

        # 读取JSON文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except UnicodeDecodeError as e:
            raise ValueError(f"JSON文件编码错误: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON文件格式错误: {e}")

        # 检查JSON文件格式
        if 'paddleocr_raw_result' not in data:
            raise ValueError(
                f"JSON文件格式不正确：缺少 'paddleocr_raw_result' 字段。\n"
                f"请使用 --save-result 参数生成的原始OCR结果文件。\n"
                f"当前文件包含的字段: {list(data.keys())}"
            )

        # 获取原图路径
        image_path = Path(data['image_path'])
        if not image_path.exists():
            raise FileNotFoundError(f"原图文件不存在: {image_path}")

        # 读取原图
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"OpenCV无法读取图片: {image_path}")
        except Exception as e:
            raise ValueError(f"读取图片时出错: {e}")

        # 创建副本用于绘制
        vis_img = img.copy()

        ocr_result = data.get('paddleocr_raw_result', {})
        if ocr_result.get('status') != 'success':
            print(f"OCR结果状态异常: {ocr_result.get('status', 'unknown')}")
            return str(image_path)

        # 绘制检测区域 (dt_polys) - 蓝色
        dt_polys = ocr_result.get('dt_polys', [])
        img_height = vis_img.shape[0]

        for i, poly in enumerate(dt_polys):
            if len(poly) >= 4:
                # 使用原始坐标（左上角原点）
                points = np.array(poly, dtype=np.int32)
                # 绘制多边形
                cv2.polylines(vis_img, [points], True, (255, 0, 0), 2)  # 蓝色
                # 添加标签
                cv2.putText(vis_img, f'D{i+1}', tuple(points[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 绘制识别区域 (rec_boxes) - 绿色，并添加识别文本
        rec_boxes = ocr_result.get('rec_boxes', [])
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])

        for i, box in enumerate(rec_boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[:4]
                # 绘制矩形
                cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 绿色

                # 添加识别文本和置信度
                if i < len(rec_texts):
                    text = rec_texts[i]
                    score = rec_scores[i] if i < len(rec_scores) else 0.0
                    label = f'R{i+1}: {text} ({score:.3f})'

                    # 计算文本位置（在矩形上方）
                    text_y = max(int(y1) - 10, 20)

                    # 绘制文本背景
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(vis_img, (int(x1), text_y - text_size[1] - 5),
                                 (int(x1) + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)

                    # 绘制文本
                    cv2.putText(vis_img, label, (int(x1) + 2, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 添加图例
        legend_y = 30
        cv2.putText(vis_img, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, 'Blue: Detection Areas (dt_polys)', (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(vis_img, 'Green: Recognition Areas (rec_boxes)', (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 添加统计信息
        stats_text = f'Detected: {len(dt_polys)}, Recognized: {len(rec_boxes)}'
        cv2.putText(vis_img, stats_text, (10, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 保存可视化结果
        output_path = json_path.parent / f"{json_path.stem}_visualization.jpg"
        cv2.imwrite(str(output_path), vis_img)

        print(f"可视化结果已保存到: {output_path}")
        print(f"检测区域数量: {len(dt_polys)}")
        print(f"识别区域数量: {len(rec_boxes)}")

        return str(output_path)




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
        # 获取图片尺寸用于几何特征计算
        img_height = img.shape[0] if img is not None and hasattr(img, 'shape') else 480
        img_width = img.shape[1] if img is not None and hasattr(img, 'shape') else 1080

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
                    dt_polys_count = len(ocr_result[0]['dt_polys']) if ocr_result and 'dt_polys' in ocr_result[0] else 0
                    rec_texts_count = len(ocr_result[0]['rec_texts']) if ocr_result and 'rec_texts' in ocr_result[0] else 0
                    print(f"OCR识别完成，检测到 {dt_polys_count} 个文本区域，成功识别 {rec_texts_count} 个文本")

                    # 显示检测失败的区域
                    if dt_polys_count > rec_texts_count:
                        print(f"⚠️  有 {dt_polys_count - rec_texts_count} 个检测区域识别失败")

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
                                # 转换为简体中文
                                simplified_text = self._convert_to_simplified(text)

                                # 获取边界框坐标
                                box_coords = box if isinstance(box, list) else box.tolist() if hasattr(box, 'tolist') else [0, 0, 100, 30]

                                text_info = {
                                    'text': text,
                                    'simplified_text': simplified_text,
                                    'score': float(score),
                                    'box': box_coords
                                }
                                result_data['texts'].append(text_info)

                                if debug_print:
                                    print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f})")
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

        # 使用智能多帧内容合并算法选择最佳文本
        if text_variants:
            final_text = self._select_best_text_from_variants(text_variants)
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

    def _select_best_text_from_variants(self, text_variants: List[str]) -> str:
        """
        智能多帧内容合并算法：从多个文本变体中选择最佳文本

        算法策略：
        1. 字符频率统计：统计每个位置上出现最多的字符
        2. 完整性优先：优先选择完整、无缺失的文本
        3. 一致性验证：确保选择的文本在多帧中都有出现
        4. 长度合理性：避免选择异常长或短的文本

        Args:
            text_variants: 多帧中收集的文本变体列表

        Returns:
            选择的最佳文本
        """
        if not text_variants:
            return ""

        if len(text_variants) == 1:
            return text_variants[0]

        # 转换为简体中文并去重
        normalized_variants = []
        original_mapping = {}  # 标准化文本 -> 原始文本列表

        for text in text_variants:
            normalized = self._convert_to_simplified(text) if text else ""
            normalized_variants.append(normalized)

            if normalized not in original_mapping:
                original_mapping[normalized] = []
            original_mapping[normalized].append(text)

        # 策略1: 如果有完全相同的文本占多数，直接选择
        text_counts = {}
        for normalized in normalized_variants:
            text_counts[normalized] = text_counts.get(normalized, 0) + 1

        # 找到出现次数最多的文本
        max_count = max(text_counts.values())
        most_frequent_texts = [text for text, count in text_counts.items() if count == max_count]

        # 如果只有一个最频繁的文本，且出现次数超过总数的50%，直接选择
        if len(most_frequent_texts) == 1 and max_count > len(text_variants) * 0.5:
            best_normalized = most_frequent_texts[0]
            # 从原始文本中选择最长的版本
            return max(original_mapping[best_normalized], key=len)

        # 策略2: 字符级别的投票算法
        if len(most_frequent_texts) > 1 or max_count <= len(text_variants) * 0.5:
            consensus_text = self._build_consensus_text(normalized_variants)
            if consensus_text:
                # 找到与共识文本最相似的原始文本
                best_original = self._find_most_similar_original(consensus_text, text_variants)
                return best_original

        # 策略3: 回退到传统方法 - 选择最长的最频繁文本
        best_normalized = max(most_frequent_texts, key=len)
        return max(original_mapping[best_normalized], key=len)

    def _build_consensus_text(self, text_variants: List[str]) -> str:
        """
        构建共识文本：通过字符级投票确定每个位置的最佳字符

        Args:
            text_variants: 标准化后的文本变体列表

        Returns:
            构建的共识文本
        """
        if not text_variants:
            return ""

        # 过滤空文本
        valid_texts = [text for text in text_variants if text.strip()]
        if not valid_texts:
            return ""

        # 找到最长的文本长度
        max_length = max(len(text) for text in valid_texts)

        consensus_chars = []

        # 对每个字符位置进行投票
        for pos in range(max_length):
            char_votes = {}

            for text in valid_texts:
                if pos < len(text):
                    char = text[pos]
                    char_votes[char] = char_votes.get(char, 0) + 1

            if char_votes:
                # 选择得票最多的字符
                best_char = max(char_votes.items(), key=lambda x: x[1])[0]
                consensus_chars.append(best_char)

        consensus_text = ''.join(consensus_chars)

        # 验证共识文本的合理性
        if self._validate_consensus_text(consensus_text, valid_texts):
            return consensus_text
        else:
            # 如果共识文本不合理，返回最长的文本
            return max(valid_texts, key=len)

    def _validate_consensus_text(self, consensus_text: str, original_texts: List[str]) -> bool:
        """
        验证共识文本是否合理

        Args:
            consensus_text: 构建的共识文本
            original_texts: 原始文本列表

        Returns:
            是否合理
        """
        if not consensus_text or not original_texts:
            return False

        # 检查共识文本与原始文本的相似度
        similarities = []
        for text in original_texts:
            similarity = smart_chinese_similarity(consensus_text, text)
            similarities.append(similarity)

        # 如果与大多数原始文本的相似度都很高，认为合理
        high_similarity_count = sum(1 for sim in similarities if sim >= 0.7)
        return high_similarity_count >= len(original_texts) * 0.6

    def _find_most_similar_original(self, consensus_text: str, original_texts: List[str]) -> str:
        """
        找到与共识文本最相似的原始文本

        Args:
            consensus_text: 共识文本
            original_texts: 原始文本列表

        Returns:
            最相似的原始文本
        """
        if not original_texts:
            return consensus_text

        best_text = original_texts[0]
        best_similarity = 0

        for text in original_texts:
            similarity = smart_chinese_similarity(consensus_text, text)
            if similarity > best_similarity:
                best_similarity = similarity
                best_text = text

        return best_text

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
            raw_ocr_results = self.ocr_frames()

            # 步骤4: 智能合并字幕段（直接处理原始OCR结果，包含空帧）
            print("正在执行智能合并...")
            segments = self.merge_subtitle_segments(raw_ocr_results)

            # 步骤4.5: 如果需要，输出原始未合并的调试文件
            if debug_raw:
                raw_output_path = output_srt_path.parent / f"{output_srt_path.stem}_raw.srt"
                print(f"\n生成原始OCR调试文件: {raw_output_path}")
                raw_segments = self.generate_raw_segments(raw_ocr_results)
                self.generate_srt(raw_segments, str(raw_output_path))
                print(f"原始段落数: {len(raw_segments)} 个")

            # 注意：不再使用 check_ocr_result，因为智能合并算法已经集成了过滤逻辑

            # 步骤6: 生成SRT文件
            self.generate_srt(segments, str(output_srt_path))

            print(f"\n处理完成！")
            print(f"SRT文件: {output_srt_path}")

            # 清理临时帧文件（可选）
            # shutil.rmtree(self.frames_dir)

        except Exception as e:
            print(f"处理失败: {e}")
            raise

    def ocr_single_image(self, image_path: str, save_result: bool = False) -> Dict:
        """
        对单张图片进行OCR识别

        Args:
            image_path: 图片文件路径
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

        print("使用完整图片进行OCR识别")
        ocr_img = img

        # 使用抽象的核心OCR识别方法
        try:
            core_result = self._ocr_image(ocr_img, debug_print=True)

            # 构建完整的结果数据
            result_data = {
                'image_path': str(image_path),
                'image_size': f"{width}x{height}",
                'texts': core_result['texts'],
                'combined_text': core_result['combined_text'],
                'raw_result': core_result['raw_result']
            }

            # 保存结果到文件
            if save_result:
                # 获取PaddleOCR原始结果（不进行任何过滤）
                raw_ocr_result = self.ocr.predict(ocr_img, use_textline_orientation=True)

                result_file = image_path.parent / f"{image_path.stem}_ocr_result.json"

                # 创建可序列化的原始结果
                save_data = {
                    'image_path': str(image_path),
                    'image_size': f"{ocr_img.shape[1]}x{ocr_img.shape[0]}",
                    'paddleocr_raw_result': self._serialize_ocr_result(raw_ocr_result)
                }

                import json
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2)

                print(f"PaddleOCR原始结果已保存到: {result_file}")

                # 同时保存简单的文本文件
                text_file = image_path.parent / f"{image_path.stem}_ocr_result.txt"
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"图片: {image_path}\n")
                    f.write(f"尺寸: {result_data['image_size']}\n")
                    f.write(f"裁剪: 否\n")
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
    group.add_argument(
        '--visualize-ocr',
        type=str,
        metavar='JSON_FILE',
        help='可视化OCR结果：根据JSON文件在原图上绘制检测和识别区域'
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
        '--save-result',
        action='store_true',
        help='单图OCR模式：保存PaddleOCR原始结果到JSON文件（不进行任何过滤）'
    )

    args = parser.parse_args()

# 创建提取器（用于OCR功能）
    extractor = VideoSubtitleExtractor(
        output_dir=args.output_dir,
        extract_fps=args.fps,
        subtitle_region_bottom=args.subtitle_bottom,
        subtitle_region_top=args.subtitle_top,
        use_gpu=True,  # 自动检测GPU
        start_time=0,
        duration=None
    )

    # 检查是否是可视化OCR结果模式
    if args.visualize_ocr:
        try:
            output_path = extractor.visualize_ocr_result(args.visualize_ocr)
            print(f"\n✓ 可视化完成: {output_path}")
        except Exception as e:
            print(f"✗ 可视化失败: {e}")
        return

    # 检查是否是单图OCR模式
    if args.ocr_image:
        print("=" * 50)
        print("单张图片OCR识别模式")
        print("=" * 50)

        try:
            # 执行单图OCR
            result = extractor.ocr_single_image(
                image_path=args.ocr_image,
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


    # 处理视频
    extractor.process_video(args.video, args.output, debug_raw=args.debug_raw)


if __name__ == "__main__":
    main()
