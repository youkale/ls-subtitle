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
            use_textline_orientation=None,  # 新版本推荐参数（原use_angle_cls）
            use_doc_orientation_classify=False,
            lang='ch',
            text_rec_score_thresh=0.8,
            text_det_box_thresh=0.7,
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

    def ocr_frames(self) -> List[Dict]:
        """
        对所有帧进行OCR识别（帧已经是裁剪后的字幕区域）

        Returns:
            识别结果列表
        """
        frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        results = []

        # 使用进度条
        for idx, frame_path in enumerate(tqdm(frame_files, desc="OCR识别进度", unit="帧")):
            # 读取图片（已经是裁剪后的字幕区域）
            img = cv2.imread(str(frame_path))
            if img is None:
                tqdm.write(f"警告: 无法读取帧 {frame_path}")
                continue

            # OCR识别 (使用PP-OCRv5模型的predict方法)
            # 现在图片已经是字幕区域了，不需要再次裁剪
            ocr_result = self.ocr.predict(img, use_textline_orientation=True)

            # 提取文本
            text_lines = []
            if ocr_result and len(ocr_result) > 0:
                # PaddleOCR 3.2.0 predict() 返回 OCRResult 对象（类字典）
                for item in ocr_result:
                    # OCRResult 是类字典对象，使用字典方式访问
                    if hasattr(item, 'get'):
                        # 尝试获取 rec_texts 和 rec_scores（复数形式）
                        texts = item.get('rec_texts')
                        scores = item.get('rec_scores')

                        if texts and scores:
                            for text, score in zip(texts, scores):
                                if score > 0.5 and text.strip():
                                    text_lines.append(text)
                        else:
                            # 尝试单数形式
                            text = item.get('rec_text', '')
                            confidence = item.get('rec_score', 0.0)
                            if confidence > 0.5 and text.strip():
                                text_lines.append(text)

                    # 兼容普通字典格式
                    elif isinstance(item, dict):
                        text = item.get('rec_text', '')
                        confidence = item.get('rec_score', 0.0)
                        if confidence > 0.5 and text.strip():
                            text_lines.append(text)

                    # 兼容列表格式 (旧版本API)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                            text = item[1][0]
                            confidence = item[1][1]
                            if confidence > 0.5 and text.strip():
                                text_lines.append(text)

            combined_text = " ".join(text_lines)

            # 计算实际时间戳（根据提取帧率）
            timestamp = idx / self.extract_fps

            # 调整时间戳，考虑开始时间偏移
            adjusted_timestamp = timestamp + self.start_time

            results.append({
                'frame_index': idx + 1,
                'timestamp': adjusted_timestamp,
                'text': combined_text
            })

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

    args = parser.parse_args()

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

    # 创建提取器
    extractor = VideoSubtitleExtractor(
        output_dir=args.output_dir,
        extract_fps=args.fps,
        subtitle_region_bottom=args.subtitle_bottom,
        subtitle_region_top=args.subtitle_top,
        use_gpu=not args.cpu,  # 如果指定 --cpu 则不使用GPU
        start_time=args.start_time,
        duration=args.duration
    )

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
    extractor.process_video(args.video, args.output)


if __name__ == "__main__":
    main()
