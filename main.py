import os
import json
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from paddleocr import PaddleOCR
import cv2
import shutil


class VideoSubtitleExtractor:
    """视频字幕提取器"""

    def __init__(self, output_dir: str = "output", extract_fps: int = 30,
                 subtitle_region_bottom: float = 0.1, subtitle_region_top: float = 0.45,
                 use_gpu: bool = False):
        """
        初始化字幕提取器

        Args:
            output_dir: 输出目录
            extract_fps: 提取帧率（每秒提取多少帧），默认30
            subtitle_region_bottom: 字幕区域底部位置（距离底部的百分比），默认0.1（10%）
            subtitle_region_top: 字幕区域顶部位置（距离底部的百分比），默认0.45（45%）
            use_gpu: 是否使用GPU加速，默认False
        """
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.extract_fps = extract_fps
        self.subtitle_region_bottom = subtitle_region_bottom
        self.subtitle_region_top = subtitle_region_top
        self.use_gpu = use_gpu

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

        # 使用 PP-OCRv5 模型
        self.ocr = PaddleOCR(
            use_textline_orientation=True,  # 新版本推荐参数（原use_angle_cls）
            lang='ch',
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
        使用ffmpeg提取视频帧

        Args:
            video_path: 视频文件路径
            fps: 视频帧率

        Returns:
            提取的帧数
        """
        print(f"正在提取视频帧（每秒{self.extract_fps}帧）...")

        # 创建帧输出目录
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # 根据参数提取指定帧率的帧
        output_pattern = str(self.frames_dir / "frame_%06d.jpg")

        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'fps={self.extract_fps}',  # 使用指定的提取帧率
            '-q:v', '2',  # 图片质量
            output_pattern,
            '-y'  # 覆盖已存在的文件
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # 统计提取的帧数
            frame_count = len(list(self.frames_dir.glob("frame_*.jpg")))
            print(f"成功提取 {frame_count} 帧")

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

    def ocr_frames(self) -> List[Dict]:
        """
        对所有帧进行OCR识别

        Returns:
            识别结果列表
        """
        print(f"正在进行OCR识别（字幕区域：底部{self.subtitle_region_bottom*100:.0f}%-{self.subtitle_region_top*100:.0f}%）...")

        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        results = []

        for idx, frame_path in enumerate(frame_files):
            # 读取图片
            img = cv2.imread(str(frame_path))
            height, width = img.shape[:2]

            # 计算字幕区域（从底部开始）
            # 例如：10%-45% 表示从底部10%到45%的区域
            bottom_y = int(height * (1 - self.subtitle_region_top))  # 顶部边界
            top_y = int(height * (1 - self.subtitle_region_bottom))  # 底部边界
            subtitle_region = img[bottom_y:top_y, :]

            # OCR识别 (使用PP-OCRv5模型的predict方法)
            ocr_result = self.ocr.predict(subtitle_region, use_textline_orientation=True)

            # 提取文本
            text_lines = []
            if ocr_result and len(ocr_result) > 0:
                # PaddleOCR 3.2.0 predict() 返回格式：
                # [{'dt_polys': [...], 'rec_text': '文本', 'rec_score': 0.xx}, ...]
                for item in ocr_result:
                    if isinstance(item, dict):
                        # 新版本格式：字典形式
                        text = item.get('rec_text', '')
                        confidence = item.get('rec_score', 0.0)
                        if confidence > 0.5 and text.strip():  # 过滤低置信度结果
                            text_lines.append(text)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        # 旧版本格式：列表形式 [[bbox, (text, confidence)], ...]
                        if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                            text = item[1][0]
                            confidence = item[1][1]
                            if confidence > 0.5 and text.strip():
                                text_lines.append(text)

            combined_text = " ".join(text_lines)

            # 计算实际时间戳（根据提取帧率）
            timestamp = idx / self.extract_fps

            results.append({
                'frame_index': idx + 1,
                'timestamp': timestamp,
                'text': combined_text
            })

            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(frame_files)} 帧")

        print(f"OCR识别完成，共处理 {len(results)} 帧")
        return results

    def merge_subtitle_segments(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        合并连续相同的字幕段

        Args:
            ocr_results: OCR识别结果

        Returns:
            合并后的字幕段
        """
        print("正在合并字幕段...")

        if not ocr_results:
            return []

        segments = []
        current_segment = None
        frame_duration = 1.0 / self.extract_fps  # 每帧的时长

        for result in ocr_results:
            text = result['text'].strip()

            # 跳过空文本
            if not text:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = None
                continue

            # 如果是新字幕段
            if current_segment is None:
                current_segment = {
                    'start_time': result['timestamp'],
                    'end_time': result['timestamp'] + frame_duration,
                    'text': text
                }
            # 如果文本相同，扩展当前段
            elif text == current_segment['text']:
                current_segment['end_time'] = result['timestamp'] + frame_duration
            # 如果文本不同，保存当前段并开始新段
            else:
                segments.append(current_segment)
                current_segment = {
                    'start_time': result['timestamp'],
                    'end_time': result['timestamp'] + frame_duration,
                    'text': text
                }

        # 添加最后一段
        if current_segment:
            segments.append(current_segment)

        print(f"合并后得到 {len(segments)} 个字幕段")
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

    def process_video(self, video_path: str, output_srt_path: str = None):
        """
        处理视频并生成SRT字幕

        Args:
            video_path: 视频文件路径
            output_srt_path: 输出SRT文件路径（可选）
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确定输出文件路径
        if output_srt_path is None:
            output_srt_path = self.output_dir / f"{video_path.stem}.srt"

        try:
            # 步骤1: 获取视频信息
            video_info = self.get_video_info(str(video_path))

            # 步骤2: 提取视频帧
            frame_count = self.extract_frames(str(video_path), video_info['fps'])

            if frame_count == 0:
                raise RuntimeError("未能提取任何视频帧")

            # 步骤3: OCR识别
            ocr_results = self.ocr_frames()

            # 步骤4: 合并字幕段
            segments = self.merge_subtitle_segments(ocr_results)

            # 步骤5: 生成SRT文件
            self.generate_srt(segments, str(output_srt_path))

            print(f"\n处理完成！")
            print(f"SRT文件: {output_srt_path}")

            # 清理临时帧文件（可选）
            # shutil.rmtree(self.frames_dir)

        except Exception as e:
            print(f"处理失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='使用PaddleOCR从视频中提取字幕并生成SRT文件'
    )
    parser.add_argument(
        'video',
        type=str,
        help='输入视频文件路径'
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
        default=0.1,
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
        help='使用GPU加速（需要安装paddlepaddle-gpu）'
    )

    args = parser.parse_args()

    # 验证字幕区域参数
    if args.subtitle_bottom < 0 or args.subtitle_bottom > 1:
        parser.error("--subtitle-bottom 必须在 0 和 1 之间")
    if args.subtitle_top < 0 or args.subtitle_top > 1:
        parser.error("--subtitle-top 必须在 0 和 1 之间")
    if args.subtitle_bottom >= args.subtitle_top:
        parser.error("--subtitle-bottom 必须小于 --subtitle-top")

    # 创建提取器
    extractor = VideoSubtitleExtractor(
        output_dir=args.output_dir,
        extract_fps=args.fps,
        subtitle_region_bottom=args.subtitle_bottom,
        subtitle_region_top=args.subtitle_top,
        use_gpu=args.gpu
    )

    # 处理视频
    extractor.process_video(args.video, args.output)


if __name__ == "__main__":
    main()
