#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🔍 验证重构后的代码                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

success_count=0
fail_count=0
total_count=0

for img in examples/*.jpg; do
    total_count=$((total_count + 1))
    filename=$(basename "$img")
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "测试 $total_count: $filename"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 运行OCR识别
    result=$(uv run python main.py --ocr-image "$img" 2>&1)
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        # 提取识别结果
        recognized=$(echo "$result" | grep "识别成功:" | sed 's/.*识别成功: "\(.*\)"/\1/')
        detected=$(echo "$result" | grep "检测到" | grep "个文本区域" | head -1)
        filtered=$(echo "$result" | grep "跳过: X轴偏离中心" | wc -l | tr -d ' ')
        adopted=$(echo "$result" | grep "✓ 采用:" | wc -l | tr -d ' ')
        
        echo "✅ 测试通过"
        echo "   $detected"
        if [ -n "$recognized" ]; then
            echo "   识别结果: \"$recognized\""
        else
            echo "   识别结果: [空]"
        fi
        echo "   X轴过滤: $filtered 个"
        echo "   采用文本: $adopted 个"
        success_count=$((success_count + 1))
    else
        echo "❌ 测试失败 (退出码: $exit_code)"
        fail_count=$((fail_count + 1))
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 测试总结"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "总计测试: $total_count 个"
echo "✅ 成功: $success_count 个"
echo "❌ 失败: $fail_count 个"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $fail_count -eq 0 ]; then
    echo "🎉 所有测试通过！重构验证成功！"
else
    echo "⚠️  有 $fail_count 个测试失败"
    exit 1
fi

