#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的字幕合并准确度测试
测试多个原始SRT文件与期望结果的对比
"""

import re
import random
from typing import List, Dict, Tuple
from pathlib import Path
from difflib import SequenceMatcher


def parse_srt_file(srt_path: str) -> List[Dict]:
    """解析 SRT 文件"""
    segments = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?:\n\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)

    for match in matches:
        index, start_str, end_str, text = match
        start_time = parse_timestamp(start_str)
        end_time = parse_timestamp(end_str)

        segments.append({
            'index': int(index),
            'start_time': start_time,
            'end_time': end_time,
            'text': text.strip()
        })

    return segments


def parse_timestamp(time_str: str) -> float:
    """将 SRT 时间戳转换为秒数"""
    time_str = time_str.replace(',', '.')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def srt_segments_to_ocr_results(segments: List[Dict], fps: int = 30) -> Dict[str, Dict]:
    """将 SRT 段落转换为模拟的 OCR 结果格式"""
    ocr_results = {}

    for seg in segments:
        start_time = seg['start_time']
        end_time = seg['end_time']
        text = seg['text']

        start_frame = int(start_time * fps)

        # 模拟不同的置信度
        base_confidence = 0.95
        if '沒' in text or '聽' in text:
            base_confidence = 0.88
        elif '没' in text or '呢' in text:
            base_confidence = 0.96

        confidence = base_confidence + random.uniform(-0.02, 0.02)
        confidence = max(0.0, min(1.0, confidence))

        frame_path = f"output/frames/frame_{start_frame:06d}.jpg"

        ocr_results[frame_path] = {
            'text': text,
            'box': [100, 50, 500, 100],
            'frame_index': start_frame,
            'items': [{
                'text': text,
                'simplified_text': text,
                'score': confidence,
                'box': [100, 50, 500, 100]
            }],
            'raw_result': [{
                'dt_polys': [[[100, 50], [500, 50], [500, 100], [100, 100]]],
                'rec_texts': [text],
                'rec_scores': [confidence],
                'rec_boxes': [[100, 50, 500, 100]]
            }]
        }

    return ocr_results


def compare_segments(merged: List[Dict], expected: List[Dict]) -> Dict:
    """对比合并结果与期望结果"""
    stats = {
        'merged_count': len(merged),
        'expected_count': len(expected),
        'exact_matches': 0,
        'partial_matches': 0,
        'mismatches': [],
        'missing': [],
        'extra': []
    }

    matched_expected_indices = set()

    for i, merged_seg in enumerate(merged):
        best_match_idx = -1
        best_similarity = 0

        for j, expected_seg in enumerate(expected):
            if j in matched_expected_indices:
                continue

            # 文本相似度
            text_sim = SequenceMatcher(None, merged_seg['text'], expected_seg['text']).ratio()

            # 时间重叠度
            merged_start = merged_seg['start_time']
            merged_end = merged_seg['end_time']
            expected_start = expected_seg['start_time']
            expected_end = expected_seg['end_time']

            overlap_start = max(merged_start, expected_start)
            overlap_end = min(merged_end, expected_end)
            overlap = max(0, overlap_end - overlap_start)

            merged_duration = merged_end - merged_start
            expected_duration = expected_end - expected_start

            if merged_duration > 0 and expected_duration > 0:
                time_sim = overlap / max(merged_duration, expected_duration)
            else:
                time_sim = 0

            # 综合相似度
            total_sim = text_sim * 0.7 + time_sim * 0.3

            if total_sim > best_similarity:
                best_similarity = total_sim
                best_match_idx = j

        if best_similarity > 0.6:
            expected_seg = expected[best_match_idx]
            matched_expected_indices.add(best_match_idx)

            # 检查是否完全匹配
            if merged_seg['text'] == expected_seg['text']:
                stats['exact_matches'] += 1
            else:
                stats['partial_matches'] += 1
                stats['mismatches'].append({
                    'merged': merged_seg,
                    'expected': expected_seg,
                    'similarity': best_similarity
                })
        else:
            stats['extra'].append(merged_seg)

    # 查找缺失的段落
    for j, expected_seg in enumerate(expected):
        if j not in matched_expected_indices:
            stats['missing'].append(expected_seg)

    return stats


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def test_single_file(raw_srt: str, expected_srt: str, fps: int = 30) -> Dict:
    """测试单个文件对"""
    from main import smart_chinese_similarity

    print(f"\n{'='*80}")
    print(f"测试文件对:")
    print(f"  原始文件: {raw_srt}")
    print(f"  期望结果: {expected_srt}")
    print(f"{'='*80}\n")

    # 1. 读取原始 SRT
    print(f"[1/5] 读取原始 SRT 文件")
    raw_segments = parse_srt_file(raw_srt)
    print(f"  ✓ 原始段落数: {len(raw_segments)}\n")

    # 2. 读取期望结果
    print(f"[2/5] 读取期望结果")
    expected_segments = parse_srt_file(expected_srt)
    print(f"  ✓ 期望段落数: {len(expected_segments)}\n")

    # 3. 转换为模拟 OCR 结果
    print(f"[3/5] 转换为模拟 OCR 结果")
    ocr_results = srt_segments_to_ocr_results(raw_segments, fps)
    print(f"  ✓ 生成 OCR 结果: {len(ocr_results)} 个帧\n")

    # 4. 创建测试提取器
    print(f"[4/5] 执行智能合并")

    class TestExtractor:
        def __init__(self, fps, start_time):
            self.extract_fps = fps
            self.start_time = start_time
            try:
                from opencc import OpenCC
                self.cc = OpenCC('t2s')
            except ImportError:
                self.cc = None

        def _convert_to_simplified(self, text):
            if self.cc and text:
                return self.cc.convert(text)
            return text

        def _check_detection_regions_validity(self, dt_polys, frame_path):
            return True

        def _select_best_text_from_variants(self, text_variants):
            if not text_variants:
                return ""

            if isinstance(text_variants[0], str):
                text_variants = [{'text': t, 'confidence': 0.95} for t in text_variants]

            if len(text_variants) == 1:
                return text_variants[0]['text']

            sorted_variants = sorted(text_variants, key=lambda x: (x.get('confidence', 0.0), len(x['text'])), reverse=True)
            return sorted_variants[0]['text']

        def _fill_segment_gaps(self, merged_segments, filtered_frames, gap_time_threshold):
            final_segments = []
            frame_map = {f['frame_index']: f for f in filtered_frames}
            gap_filled_count = 0

            for seg_idx, segment in enumerate(merged_segments):
                start_frame = segment['start_frame']
                end_frame = segment['end_frame']
                text = segment['text']

                extended_end_frame = end_frame
                next_frame_idx = end_frame + 1

                while True:
                    if seg_idx < len(merged_segments) - 1:
                        next_segment_start = merged_segments[seg_idx + 1]['start_frame']
                        if next_frame_idx >= next_segment_start:
                            break

                    if next_frame_idx not in frame_map:
                        break

                    frame_data = frame_map[next_frame_idx]

                    if (frame_data['has_detection'] and
                        not frame_data['has_text'] and
                        frame_data['is_valid_region']):

                        prev_end_time = (extended_end_frame + 1) / self.extract_fps + self.start_time
                        curr_start_time = next_frame_idx / self.extract_fps + self.start_time
                        time_gap = curr_start_time - prev_end_time

                        if abs(time_gap) < 0.001:
                            extended_end_frame = next_frame_idx
                            gap_filled_count += 1
                            next_frame_idx += 1
                        else:
                            break
                    else:
                        break

                start_time = start_frame / self.extract_fps + self.start_time
                end_time = (extended_end_frame + 1) / self.extract_fps + self.start_time

                final_segments.append({
                    'text': text,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': start_frame,
                    'end_frame': extended_end_frame
                })

            return final_segments

    test_extractor = TestExtractor(fps=fps, start_time=0)

    # 执行合并
    def merge_segments_test(ocr_results, similarity_threshold=0.8, gap_time_threshold=2.0):
        if not ocr_results:
            return []

        sorted_results = sorted(ocr_results.items(), key=lambda x: x[1]['frame_index'])

        # 步骤1：预处理
        filtered_frames = []

        for frame_path, value in sorted_results:
            frame_idx = value['frame_index']
            text = value['text'].strip() if 'text' in value else ""

            is_valid_region = True
            simplified_text = test_extractor._convert_to_simplified(text) if text else ""

            confidence = 0.95
            if 'items' in value and value['items']:
                scores = [item.get('score', 0.95) for item in value['items']]
                if scores:
                    confidence = sum(scores) / len(scores)

            filtered_frames.append({
                'frame_index': frame_idx,
                'frame_path': frame_path,
                'text': simplified_text if is_valid_region else "",
                'has_text': bool(simplified_text and is_valid_region),
                'has_detection': 'raw_result' in value and value['raw_result'] is not None,
                'is_valid_region': is_valid_region,
                'raw_result': value.get('raw_result'),
                'confidence': confidence
            })

        text_frames_count = sum(1 for f in filtered_frames if f['has_text'])

        if text_frames_count == 0:
            return []

        # 步骤2：智能相似度合并
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
                    'frame_indices': [frame_idx]
                }
                text_variants = [{'text': text, 'confidence': confidence}]
            else:
                current_end_time = (current_segment['end_frame'] + 1) / fps
                new_start_time = frame_idx / fps
                time_gap = new_start_time - current_end_time

                similarity = smart_chinese_similarity(current_segment['text'], text)

                is_time_continuous = abs(time_gap) < 0.001
                is_short_gap = 0 < time_gap < 0.15
                is_similar = similarity >= similarity_threshold

                should_merge = is_similar and (is_time_continuous or is_short_gap)

                if should_merge:
                    current_segment['end_frame'] = frame_idx
                    current_segment['frame_indices'].append(frame_idx)
                    text_variants.append({'text': text, 'confidence': confidence})
                else:
                    current_segment['text'] = test_extractor._select_best_text_from_variants(text_variants)
                    merged_segments.append(current_segment)

                    current_segment = {
                        'start_frame': frame_idx,
                        'end_frame': frame_idx,
                        'text': text,
                        'frame_indices': [frame_idx]
                    }
                    text_variants = [{'text': text, 'confidence': confidence}]

        if current_segment:
            current_segment['text'] = test_extractor._select_best_text_from_variants(text_variants)
            merged_segments.append(current_segment)

        # 步骤3：间隙填充
        final_segments = test_extractor._fill_segment_gaps(merged_segments, filtered_frames, gap_time_threshold)

        return final_segments

    merged_segments = merge_segments_test(ocr_results=ocr_results)
    print(f"  ✓ 合并完成: {len(merged_segments)} 个段落\n")

    # 5. 对比结果
    print(f"[5/5] 对比合并结果与期望结果")
    stats = compare_segments(merged_segments, expected_segments)

    return {
        'raw_file': raw_srt,
        'expected_file': expected_srt,
        'raw_count': len(raw_segments),
        'expected_count': len(expected_segments),
        'merged_count': len(merged_segments),
        'stats': stats,
        'merged_segments': merged_segments
    }


def print_test_results(result: Dict):
    """打印测试结果"""
    stats = result['stats']

    print(f"\n{'─'*80}")
    print(f"📊 测试结果统计")
    print(f"{'─'*80}")
    print(f"原始段落数: {result['raw_count']}")
    print(f"期望段落数: {result['expected_count']}")
    print(f"合并段落数: {result['merged_count']}")
    print(f"{'─'*80}")
    print(f"完全匹配: {stats['exact_matches']} 个")
    print(f"部分匹配: {stats['partial_matches']} 个")
    print(f"多余段落: {len(stats['extra'])} 个")
    print(f"缺失段落: {len(stats['missing'])} 个")
    print(f"{'─'*80}")

    total_matches = stats['exact_matches'] + stats['partial_matches']
    if result['expected_count'] > 0:
        match_rate = (total_matches / result['expected_count']) * 100
        exact_rate = (stats['exact_matches'] / result['expected_count']) * 100
        print(f"匹配率: {match_rate:.1f}% ({total_matches}/{result['expected_count']})")
        print(f"完全匹配率: {exact_rate:.1f}% ({stats['exact_matches']}/{result['expected_count']})")

    if result['raw_count'] > 0:
        compression_rate = (1 - result['merged_count'] / result['raw_count']) * 100
        print(f"压缩率: {compression_rate:.1f}% ({result['raw_count']} → {result['merged_count']})")

    # 显示不匹配的段落
    if stats['mismatches'] and len(stats['mismatches']) > 0:
        print(f"\n{'─'*80}")
        print(f"⚠️  部分匹配的段落 (前5个):")
        print(f"{'─'*80}")
        for i, mismatch in enumerate(stats['mismatches'][:5], 1):
            merged = mismatch['merged']
            expected = mismatch['expected']
            similarity = mismatch['similarity']

            print(f"\n{i}. 相似度: {similarity:.2f}")
            print(f"   合并结果: [{format_time(merged['start_time'])} → {format_time(merged['end_time'])}]")
            print(f"             \"{merged['text']}\"")
            print(f"   期望结果: [{format_time(expected['start_time'])} → {format_time(expected['end_time'])}]")
            print(f"             \"{expected['text']}\"")


def main():
    """主测试函数"""
    print(f"\n{'='*80}")
    print(f"字幕合并准确度完整测试")
    print(f"{'='*80}\n")

    # 测试文件对列表
    test_cases = [
        ('1_raw.srt', 'ghost_cut.srt'),
        ('2_raw.srt', 'ghost_cut_2.srt')
    ]

    results = []

    for raw_file, expected_file in test_cases:
        # 检查文件是否存在
        if not Path(raw_file).exists():
            print(f"❌ 文件不存在: {raw_file}")
            continue
        if not Path(expected_file).exists():
            print(f"❌ 文件不存在: {expected_file}")
            continue

        # 执行测试
        result = test_single_file(raw_file, expected_file)
        results.append(result)

        # 打印结果
        print_test_results(result)

        # 保存合并结果
        output_file = Path(raw_file).stem + '_merged_test.srt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, seg in enumerate(result['merged_segments'], 1):
                f.write(f"{idx}\n")
                f.write(f"{format_time(seg['start_time'])} --> {format_time(seg['end_time'])}\n")
                f.write(f"{seg['text']}\n")
                f.write("\n")

        print(f"\n💾 合并结果已保存: {output_file}")

    # 总结
    print(f"\n{'='*80}")
    print(f"📈 全部测试总结")
    print(f"{'='*80}")

    for i, result in enumerate(results, 1):
        stats = result['stats']
        total_matches = stats['exact_matches'] + stats['partial_matches']

        if result['expected_count'] > 0:
            match_rate = (total_matches / result['expected_count']) * 100
            exact_rate = (stats['exact_matches'] / result['expected_count']) * 100
        else:
            match_rate = 0
            exact_rate = 0

        print(f"\n测试 {i}: {Path(result['raw_file']).stem}")
        print(f"  原始段落: {result['raw_count']} → 合并段落: {result['merged_count']}")
        print(f"  匹配率: {match_rate:.1f}%  |  完全匹配率: {exact_rate:.1f}%")

        if match_rate >= 95:
            status = "✅ 优秀"
        elif match_rate >= 85:
            status = "✓ 良好"
        elif match_rate >= 70:
            status = "⚠️  一般"
        else:
            status = "❌ 需改进"

        print(f"  评价: {status}")

    print(f"\n{'='*80}")
    print(f"测试完成！")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
