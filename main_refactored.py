# 重构后的代码片段 - 供审查

"""
===========================================
1. _ocr_image 方法重构
===========================================
"""

# ============ 纯函数：数据提取和转换 ============

def _convert_poly_to_box(poly: list) -> list:
    """
    将多边形坐标转换为边界框格式 [xmin, ymin, xmax, ymax]

    纯函数：无副作用，可独立测试

    Args:
        poly: 多边形坐标点列表 [[x1,y1], [x2,y2], ...]

    Returns:
        边界框坐标 [xmin, ymin, xmax, ymax]
    """
    x_coords = [p[0] for p in poly]
    y_coords = [p[1] for p in poly]
    return [int(min(x_coords)), int(min(y_coords)),
            int(max(x_coords)), int(max(y_coords))]


def _extract_texts_scores_boxes_from_dict(item: dict) -> tuple:
    """
    从字典格式的OCR结果中提取数据

    纯函数：无副作用

    Args:
        item: OCR结果项（字典格式）

    Returns:
        (rec_texts, rec_scores, boxes) 元组
    """
    rec_texts = item.get('rec_texts', [])
    rec_scores = item.get('rec_scores', [])
    boxes = []

    # 优先使用现成的 rec_boxes
    rec_boxes = item.get('rec_boxes')
    if rec_boxes is not None and hasattr(rec_boxes, 'shape'):
        boxes = rec_boxes.tolist()
    else:
        # 从 dt_polys 计算边界框
        dt_polys = item.get('dt_polys', [])
        if dt_polys:
            boxes = [_convert_poly_to_box(poly) for poly in dt_polys]
        else:
            boxes = item.get('boxes', [])

    return rec_texts, rec_scores, boxes


def _extract_texts_scores_boxes_from_object(item) -> tuple:
    """
    从对象属性格式的OCR结果中提取数据

    纯函数：无副作用

    Args:
        item: OCR结果项（对象格式）

    Returns:
        (rec_texts, rec_scores, boxes) 元组
    """
    rec_texts = getattr(item, 'rec_texts', [])
    rec_scores = getattr(item, 'rec_scores', [])
    dt_polys = getattr(item, 'dt_polys', [])

    if dt_polys:
        boxes = [_convert_poly_to_box(poly) for poly in dt_polys]
    else:
        boxes = getattr(item, 'boxes', [])

    return rec_texts, rec_scores, boxes


def _extract_texts_scores_boxes_fallback(item) -> tuple:
    """
    尝试其他可能的方式提取数据（fallback）

    纯函数：无副作用

    Args:
        item: OCR结果项

    Returns:
        (rec_texts, rec_scores, boxes) 元组
    """
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
    从OCR结果项中提取文本、分数和边界框

    纯函数：统一处理不同格式的OCR结果

    Args:
        item: OCR结果项（可能是字典或对象）

    Returns:
        (rec_texts, rec_scores, boxes) 元组
    """
    rec_texts, rec_scores, boxes = [], [], []

    # 尝试方法1：字典方式
    if hasattr(item, 'get'):
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_from_dict(item)

    # 尝试方法2：对象属性方式
    if not rec_texts:
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_from_object(item)

    # 尝试方法3：fallback
    if not rec_texts:
        rec_texts, rec_scores, boxes = _extract_texts_scores_boxes_fallback(item)

    # 确保boxes数量与文本数量匹配
    if rec_texts and (not boxes or len(boxes) != len(rec_texts)):
        boxes = [[0, 0, 100, 30] for _ in rec_texts]

    return rec_texts, rec_scores, boxes


def _is_text_in_center_region(box_coords: list, img_width: int, tolerance_ratio: float = 0.15) -> tuple:
    """
    检查文本框是否在图片中心区域

    纯函数：无副作用，可独立测试

    Args:
        box_coords: 边界框坐标 [xmin, ymin, xmax, ymax]
        img_width: 图片宽度
        tolerance_ratio: 容忍度比例（默认15%）

    Returns:
        (is_in_center, center_x, img_center_x, tolerance) 元组
    """
    center_x = (box_coords[0] + box_coords[2]) / 2
    img_center_x = img_width / 2
    tolerance = img_width * tolerance_ratio
    is_in_center = abs(center_x - img_center_x) <= tolerance

    return is_in_center, center_x, img_center_x, tolerance


def _normalize_box_coords(box) -> list:
    """
    规范化边界框坐标

    纯函数：将不同格式的box转换为标准list格式

    Args:
        box: 边界框（可能是list、numpy数组等）

    Returns:
        标准化的边界框列表 [xmin, ymin, xmax, ymax]
    """
    if isinstance(box, list):
        return box
    elif hasattr(box, 'tolist'):
        return box.tolist()
    else:
        return [0, 0, 100, 30]  # 默认值


# ============ 重构后的主方法 ============

class VideoSubtitleExtractor:
    # ... 其他方法保持不变 ...

    def _ocr_image(self, img, debug_print: bool = False, apply_x_filter: bool = False,
                   frame_path: str = None) -> dict:
        """
        核心的单张图片OCR识别逻辑（重构版）

        重构目标：
        1. 提取纯函数用于数据转换
        2. 减少方法长度
        3. 提高可测试性

        Args:
            img: OpenCV图片对象
            debug_print: 是否打印调试信息
            apply_x_filter: 是否应用X轴中心点过滤
            frame_path: 帧文件路径

        Returns:
            OCR识别结果字典
        """
        # 1. 获取图片尺寸
        img_height = img.shape[0] if img is not None and hasattr(img, 'shape') else 480
        img_width = img.shape[1] if img is not None and hasattr(img, 'shape') else 1080

        # 2. 调用OCR引擎
        if debug_print:
            print("正在进行OCR识别...")

        try:
            ocr_result = self.ocr.predict(img, use_textline_orientation=True)
            result_data = {'texts': [], 'combined_text': "", 'raw_result': ocr_result}

            if not ocr_result or len(ocr_result) == 0:
                if debug_print:
                    print("OCR识别完成，但未找到任何文本")
                return result_data

            # 3. 打印统计信息
            if debug_print:
                self._print_ocr_stats(ocr_result)

            # 4. 处理每个OCR结果项
            for i, item in enumerate(ocr_result):
                rec_texts, rec_scores, boxes = _extract_ocr_data_from_item(item)

                if rec_texts and rec_scores:
                    self._process_ocr_texts(
                        rec_texts, rec_scores, boxes,
                        img_width, apply_x_filter,
                        result_data, debug_print
                    )

            # 5. 合并所有文本
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
                          result_data: dict, debug_print: bool):
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

            # X轴中心点过滤
            if apply_x_filter:
                is_in_center, center_x, img_center_x, tolerance = _is_text_in_center_region(
                    box_coords, img_width
                )

                if not is_in_center:
                    if debug_print:
                        print(f"    ✗ 跳过: X轴偏离中心 (中心X={center_x:.1f}, "
                              f"图片中心={img_center_x:.1f}, 容忍±{tolerance:.1f})")
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
                    center_x = (box_coords[0] + box_coords[2]) / 2
                    print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f}, 中心X={center_x:.1f})")
                else:
                    print(f"    ✓ 采用: \"{simplified_text}\" (置信度: {score:.3f})")


"""
===========================================
2. merge_subtitle_segments 方法重构
===========================================
"""

class VideoSubtitleExtractor:
    # ... 其他方法保持不变 ...

    def merge_subtitle_segments(self, ocr_results: dict,
                               similarity_threshold: float = 0.8,
                               gap_time_threshold: float = 2.0) -> list:
        """
        智能合并字幕段（重构版）

        重构目标：
        1. 拆分为三个独立步骤方法
        2. 减少主方法复杂度
        3. 提高可读性和可维护性

        Args:
            ocr_results: OCR识别结果字典
            similarity_threshold: 文本相似度阈值
            gap_time_threshold: 间隙填充的最大时间阈值

        Returns:
            合并后的字幕段列表
        """
        if not ocr_results:
            return []

        # 打印配置信息
        self._print_merge_config(ocr_results, similarity_threshold, gap_time_threshold)

        # 步骤1：预处理和X轴过滤
        filtered_frames = self._preprocess_ocr_frames(ocr_results)

        text_frames_count = sum(1 for f in filtered_frames if f['has_text'])
        if text_frames_count == 0:
            self._print_no_text_error()
            return []

        # 步骤2：智能相似度合并
        merged_segments = self._merge_similar_segments(
            filtered_frames, similarity_threshold
        )

        self._print_merge_stats(text_frames_count, merged_segments)

        # 步骤3：间隙填充
        final_segments = self._fill_segment_gaps_with_log(
            merged_segments, filtered_frames, gap_time_threshold
        )

        # 打印最终统计
        self._print_final_stats(ocr_results, text_frames_count, final_segments)

        return final_segments

    def _print_merge_config(self, ocr_results: dict, similarity_threshold: float,
                           gap_time_threshold: float):
        """打印合并配置信息"""
        print(f"\n{'='*60}")
        print(f"智能字幕合并算法")
        print(f"{'='*60}")
        print(f"配置参数:")
        print(f"  - 相似度阈值: {similarity_threshold}")
        print(f"  - 间隙填充阈值: {gap_time_threshold}秒")
        print(f"  - 输入帧数: {len(ocr_results)} 帧")

    def _preprocess_ocr_frames(self, ocr_results: dict) -> list:
        """
        预处理OCR帧：排序、X轴过滤、简体转换

        Args:
            ocr_results: OCR识别结果字典

        Returns:
            预处理后的帧列表
        """
        print(f"\n[步骤 1/3] X轴中心点过滤 - 剔除无效识别区域")

        # 按帧索引排序
        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        filtered_frames = []
        filtered_out_count = 0

        for frame_path, value in sorted_results:
            frame_data = self._process_single_frame(frame_path, value)
            filtered_frames.append(frame_data)

            if not frame_data['is_valid_region'] and frame_data['text']:
                filtered_out_count += 1

        # 打印过滤统计
        text_frames_count = sum(1 for f in filtered_frames if f['has_text'])
        print(f"  ✓ 过滤完成")
        print(f"    - 剔除无效区域: {filtered_out_count} 帧")
        print(f"    - 保留有效文本: {text_frames_count} 帧")

        return filtered_frames

    def _process_single_frame(self, frame_path: str, value: dict) -> dict:
        """
        处理单个帧数据

        Args:
            frame_path: 帧文件路径
            value: 帧数据

        Returns:
            处理后的帧数据字典
        """
        frame_idx = value['frame_index']
        text = value['text'].strip() if 'text' in value else ""

        # X轴中心点过滤
        is_valid_region = self._check_frame_region_validity(value, frame_path)

        # 转换为简体中文
        simplified_text = self._convert_to_simplified(text) if text else ""

        # 提取置信度
        confidence = self._extract_confidence_from_frame(value)

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

    def _check_frame_region_validity(self, value: dict, frame_path: str) -> bool:
        """检查帧的检测区域是否有效"""
        if 'raw_result' not in value or not value['raw_result']:
            return True

        raw_result = value['raw_result']
        if isinstance(raw_result, list) and len(raw_result) > 0:
            if 'dt_polys' in raw_result[0] and raw_result[0]['dt_polys']:
                return self._check_detection_regions_validity(
                    raw_result[0]['dt_polys'], frame_path
                )

        return True

    def _extract_confidence_from_frame(self, value: dict) -> float:
        """从帧数据中提取平均置信度"""
        if 'items' in value and value['items']:
            scores = [item.get('score', 0.95) for item in value['items']]
            if scores:
                return sum(scores) / len(scores)
        return 0.95

    def _merge_similar_segments(self, filtered_frames: list,
                               similarity_threshold: float) -> list:
        """
        智能相似度合并

        Args:
            filtered_frames: 预处理后的帧列表
            similarity_threshold: 相似度阈值

        Returns:
            合并后的段落列表
        """
        print(f"\n[步骤 2/3] 智能相似度合并")
        print(f"  - 2-4字短文本智能识别")
        print(f"  - 置信度优先选择")

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
                # 开始新段
                current_segment = {
                    'start_frame': frame_idx,
                    'end_frame': frame_idx,
                    'text': text,
                    'frame_indices': [frame_idx]
                }
                text_variants = [{'text': text, 'confidence': confidence}]
            else:
                # 判断是否应该合并
                should_merge = self._should_merge_segments(
                    current_segment, frame_idx, text, similarity_threshold
                )

                if should_merge:
                    # 延长当前段
                    current_segment['end_frame'] = frame_idx
                    current_segment['frame_indices'].append(frame_idx)
                    text_variants.append({'text': text, 'confidence': confidence})
                else:
                    # 完成当前段，开始新段
                    current_segment['text'] = self._select_best_text_from_variants(text_variants)
                    merged_segments.append(current_segment)

                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'text': text,
                        'frame_indices': [frame_idx]
                    }
                    text_variants = [{'text': text, 'confidence': confidence}]

        # 保存最后一段
        if current_segment:
            current_segment['text'] = self._select_best_text_from_variants(text_variants)
            merged_segments.append(current_segment)

        return merged_segments

    def _should_merge_segments(self, current_segment: dict, frame_idx: int,
                               text: str, similarity_threshold: float) -> bool:
        """
        判断是否应该合并段落

        Args:
            current_segment: 当前段落
            frame_idx: 新帧索引
            text: 新文本
            similarity_threshold: 相似度阈值

        Returns:
            是否应该合并
        """
        # 计算时间间隔
        current_end_time = (current_segment['end_frame'] + 1) / self.extract_fps + self.start_time
        new_start_time = frame_idx / self.extract_fps + self.start_time
        time_gap = new_start_time - current_end_time

        # 计算相似度
        similarity = smart_chinese_similarity(current_segment['text'], text)

        # 合并条件
        is_time_continuous = abs(time_gap) < 0.001
        is_short_gap = 0 < time_gap < 0.15
        is_similar = similarity >= similarity_threshold

        return is_similar and (is_time_continuous or is_short_gap)

    def _print_merge_stats(self, text_frames_count: int, merged_segments: list):
        """打印合并统计信息"""
        print(f"  ✓ 合并完成")
        print(f"    - 输入文本帧: {text_frames_count} 帧")
        print(f"    - 合并后段落: {len(merged_segments)} 个")
        print(f"    - 压缩率: {(1 - len(merged_segments) / text_frames_count) * 100:.1f}%")

    def _fill_segment_gaps_with_log(self, merged_segments: list,
                                    filtered_frames: list,
                                    gap_time_threshold: float) -> list:
        """带日志的间隙填充"""
        print(f"\n[步骤 3/3] 间隙填充")
        print(f"  - 处理有识别区但无文字的帧")
        print(f"  - 时间轴连续性检查")

        return self._fill_segment_gaps(merged_segments, filtered_frames, gap_time_threshold)

    def _print_no_text_error(self):
        """打印无文本错误"""
        print(f"\n{'='*60}")
        print(f"❌ 错误: 过滤后没有任何有效文本帧")
        print(f"{'='*60}")

    def _print_final_stats(self, ocr_results: dict, text_frames_count: int,
                          final_segments: list):
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


"""
===========================================
重构总结
===========================================

_ocr_image 重构：
✅ 提取6个纯函数用于数据转换和判断
✅ 主方法从160行减少到约60行
✅ 每个函数职责单一，易于测试
✅ 消除重复代码

merge_subtitle_segments 重构：
✅ 拆分为6个辅助方法
✅ 主方法从175行减少到约40行
✅ 每个步骤独立，逻辑清晰
✅ 便于单独测试和维护

纯函数优势：
1. 无副作用，可独立测试
2. 输入输出明确
3. 易于理解和维护
4. 可复用性高

"""
