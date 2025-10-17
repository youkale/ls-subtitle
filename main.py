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


# ============ 纯函数：OCR数据处理 ============

def _convert_poly_to_box(poly: list) -> list:
    """将多边形坐标转换为边界框格式 [xmin, ymin, xmax, ymax]"""
    x_coords = [p[0] for p in poly]
    y_coords = [p[1] for p in poly]
    return [int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords))]


def _extract_texts_scores_boxes_from_dict(item: dict) -> tuple:
    """从字典格式的OCR结果中提取数据"""
    rec_texts = item.get('rec_texts', [])
    rec_scores = item.get('rec_scores', [])
    boxes = []

    rec_boxes = item.get('rec_boxes')
    if rec_boxes is not None and hasattr(rec_boxes, 'shape'):
        boxes = rec_boxes.tolist()
    else:
        dt_polys = item.get('dt_polys', [])
        if dt_polys:
            boxes = [_convert_poly_to_box(poly) for poly in dt_polys]
        else:
            boxes = item.get('boxes', [])

    return rec_texts, rec_scores, boxes


def _extract_texts_scores_boxes_from_object(item) -> tuple:
    """从对象属性格式的OCR结果中提取数据"""
    rec_texts = getattr(item, 'rec_texts', [])
    rec_scores = getattr(item, 'rec_scores', [])
    dt_polys = getattr(item, 'dt_polys', [])

    if dt_polys:
        boxes = [_convert_poly_to_box(poly) for poly in dt_polys]
    else:
        boxes = getattr(item, 'boxes', [])

    return rec_texts, rec_scores, boxes


def _extract_texts_scores_boxes_fallback(item) -> tuple:
    """尝试其他可能的方式提取数据（fallback）"""
    if hasattr(item, 'text'):
        text = item.text
        rec_texts = [text] if text else []
        rec_scores = [getattr(item, 'score', 1.0)] if text else []
        boxes = []
    elif hasattr(item, 'texts'):
        rec_texts = item.texts
        rec_scores = getattr(item, 'scores', [1.0] * len(rec_texts))
        boxes = []
    else:
        rec_texts, rec_scores, boxes = [], [], []

    return rec_texts, rec_scores, boxes


def _extract_ocr_data_from_item(item) -> tuple:
    """
    从OCR结果项中提取文本、分数和边界框（纯函数）

    统一处理不同格式的OCR结果，提取标准化数据

    Args:
        item: OCR结果项（可能是字典或对象）

    Returns:
        (rec_texts, rec_scores, boxes) 元组
    """
    rec_texts, rec_scores, boxes = [], [], []

    if hasattr(item, 'get'):
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_from_dict(item)

    if not rec_texts:
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_from_object(item)

    if not rec_texts:
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_fallback(item)

    if rec_texts and (not boxes or len(boxes) != len(rec_texts)):
        boxes = [[0, 0, 100, 30] for _ in rec_texts]

    return rec_texts, rec_scores, boxes


def _is_text_in_center_region(box_coords: list, img_width: int, tolerance_ratio: float = 0.15) -> tuple:
    """
    检查文本框是否跨越中轴线且左右比例符合要求（纯函数）

    逻辑：
    1. 对于短文本（宽度<30%图片宽度）：使用宽松的中心点距离判断
    2. 对于长文本：文本框必须跨越图片的X轴中线，且左右比例为30:70

    短文本宽松判断原因：
    - 短文本如"董事长"（3个字）文本框较窄，容易不跨越中轴线
    - 只需要文本中心点在合理范围内（±25%）即可

    长文本严格判断原因：
    - 长文本应该居中显示，跨越中轴线
    - 左右比例30%-70%，允许OCR识别偏右

    Args:
        box_coords: 边界框坐标 [xmin, ymin, xmax, ymax]
        img_width: 图片宽度
        tolerance_ratio: 左右偏移容忍度（保留兼容性）

    Returns:
        (is_valid, center_x, center_line, left_ratio) 元组
        - is_valid: 是否通过X轴过滤
        - center_x: 文本框中心X坐标
        - center_line: 图片中轴线X坐标
        - left_ratio: 文本框左侧部分占比（短文本时为0.5）
    """
    x1, y1, x2, y2 = box_coords
    center_line = img_width / 2  # 中轴线
    center_x = (x1 + x2) / 2  # 文本框中心
    width = x2 - x1

    if width <= 0:
        return False, center_x, center_line, 0.0

    # 对短文本使用宽松的判断（宽度 < 30% 图片宽度）
    if width < img_width * 0.3:
        # 短文本：只需要中心点在合理范围内（±25%，即中间50%区域）
        tolerance = img_width * 0.25
        is_valid = abs(center_x - center_line) < tolerance
        # 返回0.5作为left_ratio，表示使用了宽松判断
        return is_valid, center_x, center_line, 0.5

    # 对长文本使用严格的跨越中轴线判断
    # 检查文本框是否跨越中轴线
    crosses_center = x1 < center_line < x2

    if not crosses_center:
        # 不跨越中轴线，直接判定为不通过
        return False, center_x, center_line, 0.0

    # 计算文本框左右比例
    left_part = center_line - x1  # 中轴线左侧的部分
    right_part = x2 - center_line  # 中轴线右侧的部分

    left_ratio = left_part / width
    right_ratio = right_part / width

    # 允许的比例范围：30%-70%
    min_ratio = 0.3
    max_ratio = 0.7

    # 左侧和右侧都要在30%-70%范围内
    is_valid = (min_ratio <= left_ratio <= max_ratio) and (min_ratio <= right_ratio <= max_ratio)

    return is_valid, center_x, center_line, left_ratio


def _normalize_box_coords(box) -> list:
    """规范化边界框坐标（纯函数）"""
    if isinstance(box, list):
        return box
    elif hasattr(box, 'tolist'):
        return box.tolist()
    else:
        return [0, 0, 100, 30]


def _is_valid_text_box_size(box_coords: list, img_width: int, img_height: int) -> tuple:
    """
    检查文本框尺寸是否在合理范围内（纯函数）

    目的: 过滤掉异常大的文本框（如全屏误检测、背景元素）

    正常字幕特征:
    - 宽度: 5% - 85% 画面宽度
    - 高度: 3% - 40% 画面高度（考虑不同分辨率和双行字幕）
    - 面积: 不超过30% 画面面积

    异常案例:
    - 文本框 [116, 0, 1080, 480] 在 1080×480 画面中
    - 占据89%宽度、100%高度 → 误检测

    Args:
        box_coords: 边界框坐标 [xmin, ymin, xmax, ymax]
        img_width: 图片宽度
        img_height: 图片高度

    Returns:
        (is_valid, width_ratio, height_ratio, area_ratio) 元组
        - is_valid: 是否通过尺寸过滤
        - width_ratio: 宽度占比
        - height_ratio: 高度占比
        - area_ratio: 面积占比
    """
    x1, y1, x2, y2 = box_coords

    # 计算文本框尺寸
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    # 计算图片面积
    img_area = img_width * img_height

    # 计算比例
    width_ratio = box_width / img_width if img_width > 0 else 0
    height_ratio = box_height / img_height if img_height > 0 else 0
    area_ratio = box_area / img_area if img_area > 0 else 0

    # 尺寸阈值
    max_width_ratio = 0.85    # 最大宽度：85%
    max_height_ratio = 0.40   # 最大高度：40%（考虑不同分辨率和双行字幕）
    max_area_ratio = 0.30     # 最大面积：30%（放宽以适应正常字幕）
    min_width_ratio = 0.05    # 最小宽度：5%（过滤噪点）
    min_height_ratio = 0.03   # 最小高度：3%（过滤噪点）

    # 检查宽度
    width_valid = min_width_ratio <= width_ratio <= max_width_ratio

    # 检查高度
    height_valid = min_height_ratio <= height_ratio <= max_height_ratio

    # 检查面积
    area_valid = area_ratio <= max_area_ratio

    # 所有条件都满足才通过
    is_valid = width_valid and height_valid and area_valid

    return is_valid, width_ratio, height_ratio, area_ratio


# ============ 文本相似度计算 ============

def smart_chinese_similarity(text1: str, text2: str) -> float:
    """
    智能中文相似度计算：
    - 去除所有标点符号后进行比较
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

    # 去除所有标点符号和空格，只保留中文字符、英文字母和数字
    import re
    import string
    # 中文标点符号
    chinese_punctuation = '，。！？；：""''（）【】《》、·—…'
    # 所有要去除的标点
    all_punctuation = string.punctuation + chinese_punctuation + ' \t\n\r'

    # 创建翻译表
    translator = str.maketrans('', '', all_punctuation)
    clean1 = text1.translate(translator)
    clean2 = text2.translate(translator)

    if not clean1 or not clean2:
        return 0.0

    # 如果去除标点后完全相同，直接返回1.0
    if clean1 == clean2:
        return 1.0

    # 只保留中文字符用于后续计算
    chinese_only1 = re.sub(r'[^\u4e00-\u9fff]', '', clean1)
    chinese_only2 = re.sub(r'[^\u4e00-\u9fff]', '', clean2)

    # 如果只有中文的部分完全相同，也返回1.0
    if chinese_only1 and chinese_only2 and chinese_only1 == chinese_only2:
        return 1.0

    # 对于2-4个汉字的特殊处理
    if 2 <= len(chinese_only1) <= 4 and 2 <= len(chinese_only2) <= 4:
        # 如果长度相同，计算匹配字符数
        if len(chinese_only1) == len(chinese_only2):
            matches = sum(1 for c1, c2 in zip(chinese_only1, chinese_only2) if c1 == c2)
            required_matches = len(chinese_only1) - 1  # 需要(长度-1)个字符匹配

            if matches >= required_matches:
                return 1.0  # 认为完全匹配
            else:
                return matches / len(chinese_only1)  # 返回实际匹配比例

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

            lcs_len = lcs_length(chinese_only1, chinese_only2)
            min_len = min(len(chinese_only1), len(chinese_only2))

            # 如果公共字符数 >= min_len - 1，认为匹配
            if lcs_len >= min_len - 1:
                return 1.0
            else:
                return lcs_len / max(len(chinese_only1), len(chinese_only2))

    # 其他情况：比较去除标点后的文本相似度
    return text_similarity(clean1, clean2)


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
            text_det_box_thresh=0.6,        # 检测阈值，适中设置
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
                # 启用X轴中心点过滤，过滤非字幕区域的干扰文字
                ocr_result = self._ocr_image(img, debug_print=debug_print, apply_x_filter=True, frame_path=str(frame_path))

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

    def _print_merge_config(self, ocr_results: Dict[str, Dict], similarity_threshold: float, gap_time_threshold: float, merge_time_threshold: float):
        """打印合并配置信息"""
        print(f"\n{'='*60}")
        print(f"智能字幕合并算法")
        print(f"{'='*60}")
        print(f"配置参数:")
        print(f"  - 相似度阈值: {similarity_threshold}")
        print(f"  - 段落合并时间阈值: {merge_time_threshold}秒（{int(merge_time_threshold * 1000)}毫秒）")
        print(f"  - 间隙填充阈值: {gap_time_threshold}秒")
        print(f"  - 输入帧数: {len(ocr_results)} 帧")

    def _process_single_frame_for_merge(self, frame_path: str, value: Dict) -> Dict:
        """处理单个帧数据用于合并"""
        frame_idx = value['frame_index']
        text = value['text'].strip() if 'text' in value else ""

        # X轴中心点过滤
        is_valid_region = True
        if 'raw_result' in value and value['raw_result']:
            raw_result = value['raw_result']
            if isinstance(raw_result, list) and len(raw_result) > 0:
                if 'dt_polys' in raw_result[0] and raw_result[0]['dt_polys']:
                    is_valid_region = self._check_detection_regions_validity(
                        raw_result[0]['dt_polys'], frame_path
                    )

        # 转换为简体中文
        simplified_text = self._convert_to_simplified(text) if text else ""

        # 提取置信度
        confidence = 0.95
        if 'items' in value and value['items']:
            scores = [item.get('score', 0.95) for item in value['items']]
            if scores:
                confidence = sum(scores) / len(scores)

        return {
            'frame_index': frame_idx,
            'frame_path': frame_path,
            'text': simplified_text if is_valid_region else "",
            'has_text': bool(simplified_text and is_valid_region),
            'has_detection': 'raw_result' in value and value['raw_result'] is not None,
            'is_valid_region': is_valid_region,
            'raw_result': value.get('raw_result'),
            'confidence': confidence
        }

    def _preprocess_ocr_frames(self, ocr_results: Dict[str, Dict]) -> List[Dict]:
        """预处理OCR帧：排序、X轴过滤、简体转换"""
        print(f"\n[步骤 1/3] X轴中心点过滤 - 剔除无效识别区域")

        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        filtered_frames = []
        filtered_out_count = 0

        for frame_path, value in sorted_results:
            frame_data = self._process_single_frame_for_merge(frame_path, value)
            filtered_frames.append(frame_data)

            if not frame_data['is_valid_region'] and value.get('text', '').strip():
                filtered_out_count += 1

        text_frames_count = sum(1 for f in filtered_frames if f['has_text'])
        print(f"  ✓ 过滤完成")
        print(f"    - 剔除无效区域: {filtered_out_count} 帧")
        print(f"    - 保留有效文本: {text_frames_count} 帧")

        return filtered_frames

    def _should_merge_segments(self, current_segment: Dict, frame_idx: int,
                               text: str, similarity_threshold: float, merge_time_threshold: float) -> bool:
        """
        判断是否应该合并段落

        Args:
            current_segment: 当前段落
            frame_idx: 新帧的索引
            text: 新帧的文本
            similarity_threshold: 相似度阈值
            merge_time_threshold: 时间间隔阈值（秒）

        Returns:
            是否应该合并
        """
        current_end_time = (current_segment['end_frame'] + 1) / self.extract_fps + self.start_time
        new_start_time = frame_idx / self.extract_fps + self.start_time
        time_gap = new_start_time - current_end_time

        similarity = smart_chinese_similarity(current_segment['text'], text)

        # 使用绝对值判断时间间隔，处理可能的负值（时间重叠）或正值（时间间隔）
        is_time_continuous = abs(time_gap) < 0.001
        is_short_gap = abs(time_gap) < merge_time_threshold
        is_similar = similarity >= similarity_threshold

        return is_similar and (is_time_continuous or is_short_gap)

    def _merge_similar_segments(self, filtered_frames: List[Dict],
                               similarity_threshold: float, merge_time_threshold: float) -> List[Dict]:
        """
        智能相似度合并（帧级别 + 段落级别）

        Args:
            filtered_frames: 过滤后的帧列表
            similarity_threshold: 相似度阈值
            merge_time_threshold: 时间间隔阈值（秒）

        Returns:
            合并后的段落列表
        """
        print(f"\n[步骤 2/3] 智能相似度合并")
        print(f"  - 2-4字短文本智能识别")
        print(f"  - 置信度优先选择")
        print(f"  - 段落级别二次合并")

        merged_segments = []
        current_segment = None
        text_variants = []

        for frame_data in filtered_frames:
            if not frame_data['has_text']:
                continue

            frame_idx = frame_data['frame_index']
            text = frame_data['text']
            confidence = frame_data.get('confidence', 0.95)

            if current_segment is None:
                current_segment = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'text': text,
                    'frame_indices': [frame_idx],
                    'text_variants': [{'text': text, 'confidence': confidence}]
                }
                text_variants = [{'text': text, 'confidence': confidence}]
            else:
                should_merge = self._should_merge_segments(
                    current_segment, frame_idx, text, similarity_threshold, merge_time_threshold
                )

                if should_merge:
                    current_segment['end_frame'] = frame_idx
                    current_segment['frame_indices'].append(frame_idx)
                    current_segment['text_variants'].append({'text': text, 'confidence': confidence})
                    text_variants.append({'text': text, 'confidence': confidence})
                else:
                    # 完成当前段落，选择最佳文本
                    current_segment['text'] = self._select_best_text_from_variants(text_variants)

                    # 尝试与已存在的最后一个段落合并（段落级别合并）
                    if merged_segments:
                        last_segment = merged_segments[-1]
                        # 检查与最后一个段落的相似度和时间连续性
                        should_merge_with_last = self._should_merge_segments(
                            last_segment, current_segment['start_frame'],
                            current_segment['text'], similarity_threshold, merge_time_threshold
                        )

                        if should_merge_with_last:
                            # 合并到最后一个段落：扩展帧范围，合并text_variants
                            last_segment['end_frame'] = current_segment['end_frame']
                            if 'frame_indices' in last_segment and 'frame_indices' in current_segment:
                                last_segment['frame_indices'].extend(current_segment['frame_indices'])
                            # 合并所有text_variants
                            if 'text_variants' in last_segment and 'text_variants' in current_segment:
                                last_segment['text_variants'].extend(current_segment['text_variants'])
                            # 重新选择最佳文本（基于所有variants）
                            last_segment['text'] = self._select_best_text_from_variants(last_segment['text_variants'])
                        else:
                            # 添加为新段落
                            merged_segments.append(current_segment)
                    else:
                        # 第一个段落，直接添加
                        merged_segments.append(current_segment)

                    # 开始新段落
                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'text': text,
                        'frame_indices': [frame_idx],
                        'text_variants': [{'text': text, 'confidence': confidence}]
                    }
                    text_variants = [{'text': text, 'confidence': confidence}]

        # 处理最后一个段落
        if current_segment:
            current_segment['text'] = self._select_best_text_from_variants(text_variants)

            # 尝试与已存在的最后一个段落合并
            if merged_segments:
                last_segment = merged_segments[-1]
                should_merge_with_last = self._should_merge_segments(
                    last_segment, current_segment['start_frame'],
                    current_segment['text'], similarity_threshold, merge_time_threshold
                )

                if should_merge_with_last:
                    last_segment['end_frame'] = current_segment['end_frame']
                    if 'frame_indices' in last_segment and 'frame_indices' in current_segment:
                        last_segment['frame_indices'].extend(current_segment['frame_indices'])
                    if 'text_variants' in last_segment and 'text_variants' in current_segment:
                        last_segment['text_variants'].extend(current_segment['text_variants'])
                    last_segment['text'] = self._select_best_text_from_variants(last_segment['text_variants'])
                else:
                    merged_segments.append(current_segment)
            else:
                merged_segments.append(current_segment)

        # 清理临时字段
        for seg in merged_segments:
            if 'text_variants' in seg:
                del seg['text_variants']

        return merged_segments

    def _print_merge_stats(self, text_frames_count: int, merged_segments: List[Dict]):
        """打印合并统计信息"""
        print(f"  ✓ 合并完成")
        print(f"    - 输入文本帧: {text_frames_count} 帧")
        print(f"    - 合并后段落: {len(merged_segments)} 个")
        print(f"    - 压缩率: {(1 - len(merged_segments) / text_frames_count) * 100:.1f}%")

    def _print_final_stats(self, ocr_results: Dict, text_frames_count: int, final_segments: List[Dict]):
        """打印最终统计信息"""
        print(f"\n{'='*60}")
        print(f"✓ 智能合并完成")
        print(f"{'='*60}")
        print(f"统计信息:")
        print(f"  - 原始OCR帧数: {len(ocr_results)} 帧")
        print(f"  - 有效文本帧数: {text_frames_count} 帧")
        print(f"  - 最终段落数量: {len(final_segments)} 个")
        print(f"  - 整体压缩率: {(1 - len(final_segments) / len(ocr_results)) * 100:.1f}%")
        print(f"{'='*60}\n")

    def merge_subtitle_segments(self, ocr_results: Dict[str, Dict], similarity_threshold: float = 0.8, gap_time_threshold: float = 2.0, merge_time_threshold: float = 0.3) -> List[Dict]:
        """
        智能合并字幕段（重构版）

        重构优化：
        - 拆分为独立的辅助方法
        - 每个步骤职责单一
        - 提高代码可读性和可维护性

        算法流程：
        1. X轴中心点过滤：剔除不在字幕区域的文本
        2. 智能相似度合并：合并相似文本（2-4个汉字只要(长度-1)个字符匹配）
        3. 间隙填充：处理有检测区但无文字的帧（严格时间连续性）

        Args:
            ocr_results: OCR识别结果字典，包含所有帧的识别信息
            similarity_threshold: 文本相似度阈值（0.0-1.0），默认0.8
            gap_time_threshold: 间隙填充的最大时间阈值（秒），默认2.0秒
            merge_time_threshold: 段落合并的最大时间间隔（秒），默认0.3秒（300毫秒）

        Returns:
            合并后的字幕段列表
        """
        if not ocr_results:
            return []

        # 打印配置信息
        self._print_merge_config(ocr_results, similarity_threshold, gap_time_threshold, merge_time_threshold)

        # 步骤1：预处理和X轴过滤
        filtered_frames = self._preprocess_ocr_frames(ocr_results)

        text_frames_count = sum(1 for f in filtered_frames if f['has_text'])
        if text_frames_count == 0:
            print(f"\n{'='*60}")
            print(f"❌ 错误: 过滤后没有任何有效文本帧")
            print(f"{'='*60}")
            return []

        # 步骤2：智能相似度合并
        merged_segments = self._merge_similar_segments(filtered_frames, similarity_threshold, merge_time_threshold)
        self._print_merge_stats(text_frames_count, merged_segments)

        # 步骤3：间隙填充
        print(f"\n[步骤 3/4] 间隙填充")
        print(f"  - 处理有识别区但无文字的帧")
        print(f"  - 时间轴连续性检查")
        gap_filled_segments = self._fill_segment_gaps(merged_segments, filtered_frames, gap_time_threshold)

        # 步骤4：后处理合并 - 合并间隙填充后相邻且相似的段落
        print(f"\n[步骤 4/4] 后处理合并")
        print(f"  - 合并相似且时间连续的段落")
        print(f"  - 去除标点符号差异")
        final_segments = self._post_process_merge_segments(gap_filled_segments, similarity_threshold, merge_time_threshold)

        # 打印最终统计
        self._print_final_stats(ocr_results, text_frames_count, final_segments)

        return final_segments

    def _post_process_merge_segments(self, segments: List[Dict], similarity_threshold: float, merge_time_threshold: float) -> List[Dict]:
        """
        后处理合并：合并间隙填充后相邻且相似的段落

        在间隙填充后，某些原本不相邻的段落可能变得时间连续，
        这时需要检查它们是否应该合并。

        合并条件：
        1. 段落时间连续或几乎连续（间隙 < merge_time_threshold秒）
        2. 文本相似（去除标点后）

        Args:
            segments: 间隙填充后的段落列表
            similarity_threshold: 相似度阈值
            merge_time_threshold: 时间间隔阈值（秒）

        Returns:
            后处理合并后的段落列表
        """
        if not segments or len(segments) <= 1:
            print(f"  ✓ 无需后处理（段落数 <= 1）")
            return segments

        merged = []
        i = 0
        merge_count = 0

        while i < len(segments):
            current = segments[i].copy()

            # 尝试与后续段落合并
            while i + 1 < len(segments):
                next_seg = segments[i + 1]

                # 计算时间间隙（使用段落的实际时间）
                current_end_time = current.get('end_time', (current['end_frame'] + 1) / self.extract_fps + self.start_time)
                next_start_time = next_seg.get('start_time', next_seg['start_frame'] / self.extract_fps + self.start_time)
                time_gap = next_start_time - current_end_time

                # 计算相似度
                similarity = smart_chinese_similarity(current['text'], next_seg['text'])

                # 判断是否应该合并
                # 时间必须连续或非常接近，且文本相似
                is_time_continuous = abs(time_gap) < merge_time_threshold
                is_similar = similarity >= similarity_threshold
                should_merge = is_time_continuous and is_similar

                if should_merge:
                    # 合并：扩展当前段落的结束帧和时间
                    current['end_frame'] = next_seg['end_frame']
                    current['end_time'] = next_seg.get('end_time', (next_seg['end_frame'] + 1) / self.extract_fps + self.start_time)

                    # 合并帧索引列表
                    if 'frame_indices' in current and 'frame_indices' in next_seg:
                        current['frame_indices'].extend(next_seg['frame_indices'])

                    # 选择更完整的文本（带标点且长度更长的优先）
                    if len(next_seg['text']) > len(current['text']):
                        current['text'] = next_seg['text']

                    merge_count += 1
                    i += 1  # 跳过已合并的段落
                else:
                    break  # 不能合并，停止

            merged.append(current)
            i += 1

        if merge_count > 0:
            print(f"  ✓ 后处理完成")
            print(f"    - 合并段落对数: {merge_count}")
            print(f"    - 合并前: {len(segments)} 个段落")
            print(f"    - 合并后: {len(merged)} 个段落")
        else:
            print(f"  ✓ 无需后处理合并（无相邻相似段落）")

        return merged

    def _fill_segment_gaps(self, merged_segments: List[Dict], filtered_frames: List[Dict], gap_time_threshold: float) -> List[Dict]:
        """
        填充段落之间的间隙 - 处理有识别区但无文字的帧

        填充策略（保守安全）：
        1. 只在时间轴完美连续时填充（前一帧结束=当前帧开始）
        2. 累计填充时长不超过gap_time_threshold（默认2.0秒，限制最多0.5秒）
        3. 考虑到下一段的距离，避免跨段填充

        场景示例：
          段落1: "弄死她" @ 02:01,333 --> 02:02,233
          帧X:   ""       @ 02:02,233 --> 02:02,533  (有检测区但无文字)
          段落2: "弄死她" @ 02:02,533 --> 02:03,533

          → 时间完美连续，填充 ✓

        Args:
            merged_segments: 已合并的字幕段列表
            filtered_frames: 所有经过预处理的帧数据列表
            gap_time_threshold: 累计填充的最大时间阈值（秒），默认2.0秒，实际限制为0.5秒

        Returns:
            填充间隙后的字幕段列表，包含时间戳
        """
        if not merged_segments:
            return []

        # 创建帧索引到帧数据的映射
        frame_map = {f['frame_index']: f for f in filtered_frames}

        # 设置累计填充的最大时长（取较小值，最多0.5秒）
        max_cumulative_fill_time = min(gap_time_threshold, 0.5)
        frame_duration = 1.0 / self.extract_fps

        # 扩展段落以填充间隙
        extended_segments = []
        gap_filled_count = 0
        segments_with_fill = 0

        for seg_idx, segment in enumerate(merged_segments):
            start_frame = segment['start_frame']
            end_frame = segment['end_frame']
            text = segment['text']

            # 尝试向后扩展（填充后续有检测区但无文字的帧）
            extended_end_frame = end_frame
            cumulative_fill_time = 0.0  # 累计填充时长
            original_end_frame = end_frame

            # 检查到下一段的总距离（如果不是最后一段）
            next_segment_distance = float('inf')
            if seg_idx < len(merged_segments) - 1:
                next_segment_start = merged_segments[seg_idx + 1]['start_frame']
                next_segment_distance = (next_segment_start - end_frame) * frame_duration

            # 检查后续帧
            next_frame_idx = end_frame + 1
            while True:
                # 保护机制：不能扩展到下一个段落的开始位置
                if seg_idx < len(merged_segments) - 1:
                    next_segment_start = merged_segments[seg_idx + 1]['start_frame']
                    if next_frame_idx >= next_segment_start:
                        break

                # 检查这一帧是否存在
                if next_frame_idx not in frame_map:
                    break

                frame_data = frame_map[next_frame_idx]

                # 检查是否有检测区但无文字，且是有效区域
                if (frame_data['has_detection'] and
                    not frame_data['has_text'] and
                    frame_data['is_valid_region']):

                    # 计算时间间隔（前一帧结尾与当前帧开头的间隔）
                    prev_end_time = (extended_end_frame + 1) / self.extract_fps + self.start_time
                    curr_start_time = next_frame_idx / self.extract_fps + self.start_time
                    time_gap = curr_start_time - prev_end_time

                    # 关键检查1：时间轴必须完美连续（误差<1ms）
                    # 如果时间不连续，说明字幕可能已经消失
                    is_time_continuous = abs(time_gap) < 0.001

                    if not is_time_continuous:
                        # 时间不连续，停止扩展
                        break

                    # 关键检查2：累计填充时长不能超过限制
                    fill_duration = frame_duration
                    if cumulative_fill_time + fill_duration > max_cumulative_fill_time:
                        # 累计填充时长超限，停止扩展
                        break

                    # 关键检查3：如果到下一段距离较远（>1秒），限制填充量
                    # 避免在长时间间隔中过度填充
                    if next_segment_distance > 1.0:
                        max_fill_for_long_gap = 0.3  # 最多填充300ms
                        if cumulative_fill_time + fill_duration > max_fill_for_long_gap:
                            break

                    # 通过所有检查，执行填充
                    extended_end_frame = next_frame_idx
                    cumulative_fill_time += fill_duration
                    gap_filled_count += 1
                    next_frame_idx += 1
                else:
                    # 这一帧有文字或没有检测区，停止扩展
                    break

            # 记录是否进行了扩展
            if extended_end_frame > original_end_frame:
                segments_with_fill += 1

            # 计算最终时间戳
            start_time = start_frame / self.extract_fps + self.start_time
            end_time = (extended_end_frame + 1) / self.extract_fps + self.start_time

            extended_segments.append({
                'text': text,
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': extended_end_frame
            })

        # 打印统计信息
        print(f"  ✓ 间隙填充完成")
        if gap_filled_count > 0:
            print(f"    - 填充帧数: {gap_filled_count} 帧")
            print(f"    - 扩展段落: {segments_with_fill} 个")
            print(f"    - 累计填充限制: {int(max_cumulative_fill_time * 1000)}ms")
            print(f"    - 时间连续性要求: 完美连接（<1ms）")
        else:
            print(f"    - 无需填充（所有段落时间不连续或文字完整）")

        return extended_segments

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

    def _convert_to_simplified(self, text: str) -> str:
        """将文本转换为简体中文"""
        if self.cc and text:
            return self.cc.convert(text)
        return text

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

        # 绘制X轴中轴线参考线
        img_height, img_width = vis_img.shape[:2]
        center_line = img_width // 2  # X轴中轴线

        # 绘制中轴线（红色实线）
        cv2.line(vis_img, (center_line, 0), (center_line, img_height), (0, 0, 255), 3)

        # 添加中轴线标注
        cv2.putText(vis_img, f'Center Line X={center_line}', (center_line + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加说明文字
        cv2.putText(vis_img, 'X-axis Filter: Text box must cross center line', (10, img_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, 'with 30%-70% on each side', (10, img_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 添加图例
        legend_y = 50
        cv2.putText(vis_img, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, 'Blue: Detection Areas (dt_polys)', (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(vis_img, 'Green: Recognition Areas (rec_boxes)', (10, legend_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(vis_img, 'Red: Center Line (subtitle area)', (10, legend_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 添加统计信息
        stats_text = f'Detected: {len(dt_polys)}, Recognized: {len(rec_boxes)}'
        cv2.putText(vis_img, stats_text, (10, legend_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 保存可视化结果
        output_path = json_path.parent / f"{json_path.stem}_visualization.jpg"
        cv2.imwrite(str(output_path), vis_img)

        print(f"可视化结果已保存到: {output_path}")
        print(f"检测区域数量: {len(dt_polys)}")
        print(f"识别区域数量: {len(rec_boxes)}")

        return str(output_path)




    def _ocr_image(self, img, debug_print: bool = False, apply_x_filter: bool = False, frame_path: str = None) -> Dict:
        """
        核心的单张图片OCR识别逻辑（重构版）

        重构优化：
        - 使用纯函数提取数据
        - 拆分为多个职责单一的辅助方法
        - 减少代码长度和复杂度

        Args:
            img: OpenCV图片对象 (numpy.ndarray)
            debug_print: 是否打印调试信息
            apply_x_filter: 是否应用X轴中心点过滤（过滤非字幕区域）
            frame_path: 帧文件路径（用于X轴过滤时获取图片尺寸）

        Returns:
            Dict: 包含识别结果的字典
        """
        # 获取图片尺寸
        img_height = img.shape[0] if img is not None and hasattr(img, 'shape') else 480
        img_width = img.shape[1] if img is not None and hasattr(img, 'shape') else 1080

        # 进行OCR识别
        if debug_print:
            print("正在进行OCR识别...")

        try:
            ocr_result = self.ocr.predict(img, use_textline_orientation=True)
            result_data = {'texts': [], 'combined_text': "", 'raw_result': ocr_result}

            if not ocr_result or len(ocr_result) == 0:
                if debug_print:
                    print("OCR识别完成，但未找到任何文本")
                return result_data

            # 打印统计信息
            if debug_print:
                self._print_ocr_stats(ocr_result)

            # 处理每个OCR结果项
            for i, item in enumerate(ocr_result):
                rec_texts, rec_scores, boxes = _extract_ocr_data_from_item(item)

                if rec_texts and rec_scores:
                    self._process_ocr_texts(
                        rec_texts, rec_scores, boxes,
                        img_width, apply_x_filter,
                        result_data, debug_print, img_height
                    )

            # 合并所有文本
            if result_data['texts']:
                combined_text = ''.join([item['simplified_text'] for item in result_data['texts']])
                result_data['combined_text'] = ''.join(combined_text.split())

                if debug_print:
                    print(f"\n合并文本: \"{result_data['combined_text']}\"")
            else:
                if debug_print:
                    print("\n未识别到任何文本")

            return result_data

        except Exception as e:
            if debug_print:
                print(f"OCR识别失败: {e}")
            raise

    def _print_ocr_stats(self, ocr_result: list):
        """打印OCR统计信息"""
        dt_polys_count = len(ocr_result[0]['dt_polys']) if ocr_result and 'dt_polys' in ocr_result[0] else 0
        rec_texts_count = len(ocr_result[0]['rec_texts']) if ocr_result and 'rec_texts' in ocr_result[0] else 0
        print(f"OCR识别完成，检测到 {dt_polys_count} 个文本区域，成功识别 {rec_texts_count} 个文本")

        if dt_polys_count > rec_texts_count:
            print(f"⚠️  有 {dt_polys_count - rec_texts_count} 个检测区域识别失败")

    def _process_ocr_texts(self, rec_texts: list, rec_scores: list, boxes: list,
                          img_width: int, apply_x_filter: bool,
                          result_data: dict, debug_print: bool, img_height: int = None):
        """处理OCR识别的文本列表"""
        for j, (text, score, box) in enumerate(zip(rec_texts, rec_scores, boxes)):
            if debug_print:
                print(f"  检测到文本 {j+1}: \"{text}\" (置信度: {score:.3f})")

            if score <= 0.5:  # 置信度阈值
                if debug_print:
                    print(f"    ✗ 跳过: 置信度过低 ({score:.3f} < 0.5)")
                continue

            # 规范化边界框
            box_coords = _normalize_box_coords(box)

            # 几何尺寸过滤（优先检查，过滤掉异常大的文本框）
            if apply_x_filter and img_height:
                size_valid, width_ratio, height_ratio, area_ratio = _is_valid_text_box_size(
                    box_coords, img_width, img_height
                )

                if not size_valid:
                    if debug_print:
                        x1, y1, x2, y2 = box_coords
                        box_width = x2 - x1
                        box_height = y2 - y1
                        print(f"    ✗ 跳过: 文本框尺寸异常")
                        print(f"       文本框: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                        print(f"       尺寸: {box_width:.0f}×{box_height:.0f}px")
                        print(f"       宽度占比: {width_ratio:.1%} (限制: 5%-85%)")
                        print(f"       高度占比: {height_ratio:.1%} (限制: 3%-40%)")
                        print(f"       面积占比: {area_ratio:.1%} (限制: ≤30%)")
                    continue

            # X轴中心点过滤
            if apply_x_filter:
                is_valid, center_x, center_line, left_ratio = _is_text_in_center_region(
                    box_coords, img_width
                )

                if not is_valid:
                    if debug_print:
                        x1, y1, x2, y2 = box_coords
                        right_ratio = 1 - left_ratio if left_ratio > 0 else 0
                        crosses = "是" if x1 < center_line < x2 else "否"
                        print(f"    ✗ 跳过: X轴过滤未通过")
                        print(f"       文本框: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                        print(f"       中轴线: X={center_line:.0f}")
                        print(f"       是否跨越中轴线: {crosses}")
                        if left_ratio > 0:
                            print(f"       左右比例: {left_ratio:.1%} : {right_ratio:.1%} (要求: 30%-70%)")
                    continue

            # 转换为简体中文并保存
            simplified_text = self._convert_to_simplified(text)
            text_info = {
                'text': text,
                'simplified_text': simplified_text,
                'score': float(score),
                'box': box_coords
            }
            result_data['texts'].append(text_info)

            # 打印采用信息
            if debug_print:
                if apply_x_filter:
                    x1, y1, x2, y2 = box_coords
                    center_x = (x1 + x2) / 2
                    width = x2 - x1
                    center_line = img_width / 2
                    left_part = center_line - x1
                    left_ratio = left_part / width if width > 0 else 0
                    right_ratio = 1 - left_ratio
                    print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f})")
                    print(f"       文本框跨越中轴线, 左右比例: {left_ratio:.1%} : {right_ratio:.1%}")
                else:
                    print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f})")


    def _select_best_text_from_variants(self, text_variants: List) -> str:
        """
        智能多帧内容合并算法：从多个文本变体中选择最佳文本

        算法策略（按优先级）：
        1. 置信度优先：优先选择高置信度的文本
        2. 字符频率统计：统计每个位置上出现最多的字符
        3. 完整性优先：优先选择完整、无缺失的文本
        4. 一致性验证：确保选择的文本在多帧中都有出现
        5. 长度合理性：避免选择异常长或短的文本

        Args:
            text_variants: 多帧中收集的文本变体列表
                         可以是字符串列表（兼容旧版）或字典列表 [{'text': str, 'confidence': float}]

        Returns:
            选择的最佳文本
        """
        if not text_variants:
            return ""

        # 兼容旧版：如果是字符串列表，转换为字典列表
        if isinstance(text_variants[0], str):
            text_variants = [{'text': t, 'confidence': 0.95} for t in text_variants]

        if len(text_variants) == 1:
            return text_variants[0]['text']

        # 策略1: 置信度优先 - 如果有明显高置信度的文本，优先选择
        # 按置信度排序
        sorted_by_confidence = sorted(text_variants, key=lambda x: x.get('confidence', 0.0), reverse=True)
        highest_confidence = sorted_by_confidence[0].get('confidence', 0.0)

        # 如果最高置信度明显高于其他（差距>0.1），且不是最短的，直接选择
        if len(sorted_by_confidence) > 1:
            second_confidence = sorted_by_confidence[1].get('confidence', 0.0)
            if highest_confidence - second_confidence > 0.1:
                # 额外检查：如果最高置信度的文本明显更短，可能不完整
                # 在这种情况下，考虑次高置信度的更长文本
                highest_text = sorted_by_confidence[0]['text']
                second_text = sorted_by_confidence[1]['text']
                # 如果最高置信度文本明显短于第二名（少30%以上），选择第二名
                if len(highest_text) > 0 and len(second_text) > len(highest_text) * 1.3:
                    # 差距不是特别大，且第二名更完整，选择第二名
                    if highest_confidence - second_confidence < 0.15:
                        return second_text
                return highest_text

        # 收集所有高置信度的文本（置信度在最高值的0.05范围内）
        high_confidence_variants = [v for v in text_variants if v.get('confidence', 0.0) >= highest_confidence - 0.05]

        # 如果只有一个高置信度文本，直接返回
        if len(high_confidence_variants) == 1:
            return high_confidence_variants[0]['text']

        # 如果有多个高置信度文本（置信度相近），优先选择最长的
        # 这样可以避免选择不完整的文本（如"带" vs "带孩子"）
        if len(high_confidence_variants) > 1:
            # 先按置信度和长度综合排序
            # 优先级：置信度>长度
            best_variant = max(high_confidence_variants,
                             key=lambda x: (x.get('confidence', 0.0), len(x['text'])))
            # 检查是否有其他文本长度明显更长且置信度相近
            for v in high_confidence_variants:
                if (len(v['text']) > len(best_variant['text']) * 1.2 and
                    v.get('confidence', 0.0) >= best_variant.get('confidence', 0.0) - 0.03):
                    best_variant = v
            return best_variant['text']

        # 转换为简体中文并去重（只处理高置信度的文本）
        normalized_variants = []
        original_mapping = {}  # 标准化文本 -> (原始文本, 置信度) 列表
        confidence_mapping = {}  # 标准化文本 -> 平均置信度

        for variant in high_confidence_variants:
            text = variant['text']
            confidence = variant.get('confidence', 0.95)
            normalized = self._convert_to_simplified(text) if text else ""
            normalized_variants.append(normalized)

            if normalized not in original_mapping:
                original_mapping[normalized] = []
                confidence_mapping[normalized] = []
            original_mapping[normalized].append(text)
            confidence_mapping[normalized].append(confidence)

        # 计算每个标准化文本的平均置信度
        avg_confidence = {
            text: sum(scores) / len(scores)
            for text, scores in confidence_mapping.items()
        }

        # 策略2: 如果有完全相同的文本占多数，选择置信度最高的那个
        text_counts = {}
        for normalized in normalized_variants:
            text_counts[normalized] = text_counts.get(normalized, 0) + 1

        # 找到出现次数最多的文本
        max_count = max(text_counts.values())
        most_frequent_texts = [text for text, count in text_counts.items() if count == max_count]

        # 如果只有一个最频繁的文本，且出现次数超过总数的50%
        if len(most_frequent_texts) == 1 and max_count > len(text_variants) * 0.5:
            best_normalized = most_frequent_texts[0]
            # 从原始文本中选择置信度最高的版本
            texts_with_conf = list(zip(original_mapping[best_normalized], confidence_mapping[best_normalized]))
            return max(texts_with_conf, key=lambda x: (x[1], len(x[0])))[0]

        # 如果有多个最频繁的文本，选择平均置信度最高的
        if len(most_frequent_texts) > 1:
            best_normalized = max(most_frequent_texts, key=lambda t: (avg_confidence[t], len(t)))
            texts_with_conf = list(zip(original_mapping[best_normalized], confidence_mapping[best_normalized]))
            return max(texts_with_conf, key=lambda x: (x[1], len(x[0])))[0]

        # 策略3: 字符级别的投票算法
        if max_count <= len(text_variants) * 0.5:
            consensus_text = self._build_consensus_text(normalized_variants)
            if consensus_text:
                # 找到与共识文本最相似且置信度高的原始文本
                all_texts = [v['text'] for v in high_confidence_variants]
                best_original = self._find_most_similar_original(consensus_text, all_texts)
                return best_original

        # 策略4: 回退 - 选择置信度最高的文本
        return sorted_by_confidence[0]['text']

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
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, segment in enumerate(segments, 1):
                start_time = self.format_timestamp(segment['start_time'])
                end_time = self.format_timestamp(segment['end_time'])
                text = segment['text']

                f.write(f"{idx}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{text}\n")
                f.write("\n")

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

        print(f"\n{'='*80}")
        print(f"视频字幕提取流程")
        print(f"{'='*80}")
        print(f"输入视频: {video_path.name}")
        print(f"输出文件: {output_srt_path}")
        print(f"{'='*80}\n")

        try:
            # 步骤1: 获取视频信息
            print(f"[1/4] 获取视频信息")
            video_info = self.get_video_info(str(video_path))
            print(f"  ✓ 视频信息获取完成\n")

            # 步骤2: 提取视频帧
            print(f"[2/4] 提取视频帧")
            frame_count = self.extract_frames(str(video_path), video_info['fps'])

            if frame_count == 0:
                raise RuntimeError("未能提取任何视频帧")
            print(f"  ✓ 帧提取完成\n")

            # 步骤3: OCR识别
            print(f"[3/4] OCR文字识别")
            raw_ocr_results = self.ocr_frames()
            print(f"  ✓ OCR识别完成\n")

            # 步骤4: 智能合并字幕段
            print(f"[4/4] 智能合并处理")
            segments = self.merge_subtitle_segments(raw_ocr_results)

            # 步骤4.5: 如果需要，输出原始未合并的调试文件
            if debug_raw:
                raw_output_path = output_srt_path.parent / f"{output_srt_path.stem}_raw.srt"
                print(f"\n[调试] 生成原始OCR调试文件")
                raw_segments = self.generate_raw_segments(raw_ocr_results)
                self.generate_srt(raw_segments, str(raw_output_path))
                print(f"  ✓ 调试文件: {raw_output_path}")
                print(f"  ✓ 原始段落数: {len(raw_segments)} 个\n")

            # 步骤5: 生成SRT文件
            self.generate_srt(segments, str(output_srt_path))

            print(f"{'='*80}")
            print(f"✓ 处理完成！")
            print(f"{'='*80}")
            print(f"输出文件: {output_srt_path}")
            print(f"段落数量: {len(segments)} 个")
            print(f"{'='*80}\n")

            # 清理临时帧文件（可选）
            # shutil.rmtree(self.frames_dir)

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ 处理失败")
            print(f"{'='*80}")
            print(f"错误信息: {e}")
            print(f"{'='*80}\n")
            raise

    def ocr_single_image(self, image_path: str) -> Dict:
        """
        对单张图片进行OCR识别

        默认会：
        1. 保存OCR原始结果到JSON文件
        2. 保存简单文本结果到TXT文件
        3. 生成可视化图片（带检测框和识别结果）
        4. 应用X轴中心点过滤（仅保留字幕区域文本）

        Args:
            image_path: 图片文件路径

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
        print("✓ 已启用X轴中心点过滤（仅保留字幕区域文本）")
        ocr_img = img

        # 使用抽象的核心OCR识别方法（默认启用X轴过滤）
        try:
            core_result = self._ocr_image(ocr_img, debug_print=True, apply_x_filter=True, frame_path=str(image_path))

            # 构建完整的结果数据
            result_data = {
                'image_path': str(image_path),
                'image_size': f"{width}x{height}",
                'texts': core_result['texts'],
                'combined_text': core_result['combined_text'],
                'raw_result': core_result['raw_result']
            }

            # 默认保存结果到文件
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

            # 自动生成可视化图片
            try:
                vis_output = self.visualize_ocr_result(str(result_file))
                print(f"可视化结果已保存到: {vis_output}")
            except Exception as e:
                print(f"⚠️  可视化生成失败: {e}")

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
            # 执行单图OCR（默认保存结果、生成可视化、应用X轴过滤）
            result = extractor.ocr_single_image(image_path=args.ocr_image)

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
