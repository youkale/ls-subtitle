# LS-Subtitle - 视频字幕识别工具

使用PaddleOCR (PP-OCRv5模型) 从视频中提取字幕并生成SRT文件。

## 功能特性

- 自动检测视频帧率
- 使用ffmpeg提取视频帧
- 使用PaddleOCR PP-OCRv5模型识别字幕文本（更高精度）
- 自动合并连续相同的字幕段
- 生成标准SRT字幕文件
- 支持自定义提取帧率和字幕区域

## 安装依赖

### 系统依赖

首先需要安装 ffmpeg 和 ffprobe：

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows:**
下载并安装 [ffmpeg](https://ffmpeg.org/download.html)

### Python依赖

**CPU版本（默认）：**

使用 uv 安装Python依赖：

```bash
uv sync
```

或使用 pip：

```bash
pip install -e .
```

**GPU版本（CUDA加速）：**

如果你有NVIDIA GPU和CUDA环境，可以安装GPU版本以获得更快的处理速度：

使用 uv：

```bash
# 先卸载CPU版本的paddlepaddle
pip uninstall paddlepaddle

# 安装GPU依赖
uv sync --extra gpu
```

或使用 pip：

```bash
# 先卸载CPU版本的paddlepaddle
pip uninstall paddlepaddle

# 安装GPU版本
pip install -e ".[gpu]"
```

**注意**: GPU版本需要：
- NVIDIA GPU
- CUDA 11.2+ 或 CUDA 12.0+
- cuDNN 8.2+

## 使用方法

### 基本用法

```bash
python main.py <视频文件路径>
```

### 指定输出文件

```bash
python main.py <视频文件路径> -o output.srt
```

### 指定输出目录

```bash
python main.py <视频文件路径> --output-dir ./my_output
```

### 自定义提取帧率

```bash
# 每秒提取30帧（默认值）
python main.py video.mp4 --fps 30

# 每秒提取1帧（更快，但可能漏掉部分字幕）
python main.py video.mp4 --fps 1
```

### 自定义字幕区域

```bash
# 识别底部10%-45%区域的字幕（默认值）
python main.py video.mp4 --subtitle-bottom 0.1 --subtitle-top 0.45

# 识别底部15%-35%区域的字幕
python main.py video.mp4 --subtitle-bottom 0.15 --subtitle-top 0.35
```

### 使用GPU加速

```bash
# 使用GPU加速处理（需要安装GPU版本）
python main.py video.mp4 --gpu

# 结合其他参数使用GPU
python main.py video.mp4 --gpu --fps 30 --subtitle-bottom 0.1 --subtitle-top 0.45
```

### 完整示例

```bash
# 处理视频文件，使用默认参数（CPU模式）
python main.py video.mp4

# 处理视频文件，使用GPU加速
python main.py video.mp4 --gpu

# 处理视频文件，指定所有参数
python main.py video.mp4 -o output.srt --fps 25 --subtitle-bottom 0.1 --subtitle-top 0.4

# GPU加速 + 自定义参数
python main.py video.mp4 --gpu --fps 30 -o output.srt

# 输出将保存在 output/video.srt
# 提取的帧会保存在 output/frames/ 目录
```

## 工作原理

1. **视频信息提取**: 使用 `ffprobe` 获取视频的帧率、时长和分辨率
2. **帧提取**: 使用 `ffmpeg` 按指定帧率提取图片（默认每秒30帧）
3. **OCR识别**: 使用 PaddleOCR PP-OCRv5 模型识别视频底部指定区域的文字（默认底部10%-45%）
4. **字幕合并**: 合并连续相同的字幕段
5. **SRT生成**: 生成标准格式的SRT字幕文件

## 参数说明

### --fps (提取帧率)
- **类型**: 整数
- **默认值**: 30
- **说明**: 每秒提取多少帧进行OCR识别
- **建议**:
  - 高帧率（25-30）：识别更准确，但处理时间更长
  - 低帧率（1-5）：处理更快，但可能遗漏部分字幕
  - 视频字幕变化较慢时可使用较低帧率

### --subtitle-bottom (字幕区域底部)
- **类型**: 浮点数 (0-1)
- **默认值**: 0.1 (10%)
- **说明**: 字幕区域距离视频底部的百分比
- **示例**: 0.1 表示从底部向上10%的位置

### --subtitle-top (字幕区域顶部)
- **类型**: 浮点数 (0-1)
- **默认值**: 0.45 (45%)
- **说明**: 字幕区域距离视频底部的百分比
- **示例**: 0.45 表示从底部向上45%的位置
- **注意**: subtitle-top 必须大于 subtitle-bottom

### --gpu (GPU加速)
- **类型**: 布尔标志
- **默认值**: False (使用CPU)
- **说明**: 启用GPU加速进行OCR识别
- **要求**:
  - 需要安装 paddlepaddle-gpu
  - NVIDIA GPU + CUDA环境
  - 程序会自动检测GPU可用性
- **性能**: GPU加速可以提升3-5倍的处理速度

### 字幕区域示例

```
视频顶部 ─────────────────
          ▲
          │ (未识别区域)
          │
45% ─────────────────  ← subtitle-top
          ▲
          │ (字幕识别区域)
          │ 这是要识别的区域
          │
10% ─────────────────  ← subtitle-bottom
          │ (未识别区域)
          │
视频底部 ─────────────────
```

## 高级配置

如需修改其他参数，可以在代码中调整：

- **OCR置信度阈值**: 在 `ocr_frames` 方法中修改 `confidence > 0.5`（默认0.5）
- **OCR语言**: 修改 `PaddleOCR(lang='ch')` 参数支持其他语言
- **GPU设备选择**: 修改 `device='gpu:0'` 来指定不同的GPU（如 `gpu:1` 使用第2块GPU）
- **多GPU并行**: PaddleOCR 3.2.0+ 支持通过 `device` 参数灵活指定计算设备

## 注意事项

- 首次运行会自动下载 PaddleOCR PP-OCRv5 模型文件（约100MB）
- PP-OCRv5 是最新版本的OCR模型，提供更高的识别精度
- 处理时间取决于视频长度和提取帧率：
  - CPU模式：通常1分钟视频需要1-2分钟处理时间
  - GPU模式：可以提升3-5倍速度，1分钟视频约20-40秒
- 默认识别中文字幕，如需其他语言，修改 `PaddleOCR(lang='ch')` 参数
- 提取的帧图片会保存在 `output/frames/` 目录，可以手动删除
- 建议首次运行时使用较短的测试视频，以验证配置是否正确
- PaddleOCR 3.2.0+ 使用 `device` 参数控制计算设备（`cpu` 或 `gpu:0`）

## 输出格式

生成的SRT文件格式如下：

```
1
00:00:01,000 --> 00:00:03,000
这是第一句字幕

2
00:00:04,000 --> 00:00:06,000
这是第二句字幕
```

## 故障排除

### 问题：找不到 ffmpeg 或 ffprobe

确保已安装 ffmpeg 并添加到系统 PATH 中。

### 问题：OCR识别效果不好

- 确保视频清晰度足够
- 字幕文字对比度要高
- 可以调整字幕区域范围
- 可以降低置信度阈值

### 问题：处理速度慢

- 可以减少提取帧率（如 `--fps 1` 每秒提取1帧）
- **推荐**: 使用GPU加速（`--gpu` 参数），速度提升3-5倍
- 安装GPU版本: `pip uninstall paddlepaddle && pip install -e ".[gpu]"`

### 问题：GPU不可用

- 确认已安装 `paddlepaddle-gpu` 而非 `paddlepaddle`
- 检查CUDA是否正确安装：运行 `nvidia-smi`
- 确认CUDA版本与paddlepaddle-gpu版本兼容
- 如果GPU检测失败，程序会自动降级到CPU模式

## License

MIT
