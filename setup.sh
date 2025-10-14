#!/bin/bash
# 自动安装脚本 - 根据平台选择正确的PaddlePaddle版本

set -e

echo "=========================================="
echo "LS-Subtitle 自动安装脚本"
echo "=========================================="
echo ""

# 检测平台
OS_TYPE=$(uname -s)
echo "检测到操作系统: $OS_TYPE"

# 检测Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
echo "Python版本: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]); then
    echo "✗ 错误: 需要Python 3.12或更高版本"
    exit 1
fi

echo ""

# 根据平台安装
if [ "$OS_TYPE" = "Darwin" ]; then
    # macOS - 只支持CPU
    echo "平台: macOS"
    echo "安装模式: CPU (macOS不支持NVIDIA GPU)"
    echo ""

    # 安装基础依赖
    echo "安装依赖..."
    if command -v uv &> /dev/null; then
        uv sync
        echo "安装PaddlePaddle CPU版本..."
        uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    else
        pip install paddleocr opencv-python pillow numpy
        echo "安装PaddlePaddle CPU版本..."
        pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    fi

    echo ""
    echo "✓ 安装完成 (CPU模式)"
    echo ""
    echo "使用方法:"
    echo "  python main.py video.mp4"
    echo ""

elif [ "$OS_TYPE" = "Linux" ]; then
    # Linux - 检测是否有CUDA
    echo "平台: Linux"
    echo ""

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo "✓ 检测到CUDA: $CUDA_VERSION"
        echo ""

        # 询问用户是否安装GPU版本
        read -p "是否安装GPU版本？(y/n) [默认: y]: " INSTALL_GPU
        INSTALL_GPU=${INSTALL_GPU:-y}

        if [ "$INSTALL_GPU" = "y" ] || [ "$INSTALL_GPU" = "Y" ]; then
            echo ""
            echo "安装模式: GPU (CUDA 12.6)"
            echo ""

            # 安装基础依赖（不含paddlepaddle）
            echo "安装基础依赖..."
            if command -v uv &> /dev/null; then
                uv sync
            else
                pip install paddleocr opencv-python pillow numpy
            fi

            # 卸载可能存在的CPU版本
            pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true

            # 安装GPU版本 (CUDA 12.6)
            echo ""
            echo "安装PaddlePaddle GPU版本 (CUDA 12.6)..."
            if command -v uv &> /dev/null; then
                uv pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ || {
                    echo "✗ GPU版本安装失败，回退到CPU版本"
                    uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
                }
            else
                pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ || {
                    echo "✗ GPU版本安装失败，回退到CPU版本"
                    pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
                }
            fi

            echo ""
            echo "✓ 安装完成 (GPU模式)"
            echo ""
            echo "使用方法:"
            echo "  python main.py video.mp4 --gpu"
            echo ""
        else
            echo ""
            echo "安装模式: CPU"
            if command -v uv &> /dev/null; then
                uv sync
                uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
            else
                pip install paddleocr opencv-python pillow numpy
                pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
            fi
            echo "✓ 安装完成 (CPU模式)"
            echo ""
            echo "使用方法:"
            echo "  python main.py video.mp4"
            echo ""
        fi
    else
        echo "⚠ 未检测到CUDA"
        echo "安装模式: CPU"
        echo ""

        if command -v uv &> /dev/null; then
            uv sync
            uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
        else
            pip install paddleocr opencv-python pillow numpy
            pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
        fi

        echo "✓ 安装完成 (CPU模式)"
        echo ""
        echo "使用方法:"
        echo "  python main.py video.mp4"
        echo ""
    fi
else
    echo "⚠ 未知平台: $OS_TYPE"
    echo "安装基础依赖..."
    if command -v uv &> /dev/null; then
        uv sync
        uv pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    else
        pip install paddleocr opencv-python pillow numpy
        pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    fi
    echo ""
fi

# 验证安装
echo "=========================================="
echo "验证安装"
echo "=========================================="

python -c "
import sys
import paddle
import paddleocr

print(f'✓ Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')
print(f'✓ PaddlePaddle: {paddle.__version__}')
print(f'✓ PaddleOCR: {paddleocr.__version__}')
print(f'✓ CUDA编译支持: {paddle.device.is_compiled_with_cuda()}')

if paddle.device.is_compiled_with_cuda():
    try:
        gpu_count = paddle.device.cuda.device_count()
        print(f'✓ GPU数量: {gpu_count}')
        for i in range(gpu_count):
            try:
                gpu_name = paddle.device.cuda.get_device_name(i)
                print(f'  GPU {i}: {gpu_name}')
            except:
                print(f'  GPU {i}: 可用')
        print('')
        print('🚀 GPU加速已启用！')
    except:
        print('⚠ GPU设备初始化失败，将使用CPU模式')
        print('')
        print('💻 使用CPU模式')
else:
    print('')
    print('💻 使用CPU模式')
" || {
    echo ""
    echo "⚠ 验证失败，请检查安装"
    exit 1
}

echo ""
echo "=========================================="
echo "安装成功！"
echo "=========================================="
