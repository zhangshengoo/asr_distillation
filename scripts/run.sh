#!/bin/bash

# =============================================================================
# ASR Distillation Framework 启动脚本
# =============================================================================
# 
# 功能描述：
#   这是ASR蒸馏框架的主要启动脚本，用于简化框架的使用和管理。
#   支持运行pipeline、创建配置文件等核心操作。
#
# 使用环境要求：
#   - Python 3.9+
#   - CUDA 11.0+ (GPU推理)
#   - 足够的磁盘空间用于缓存和日志
#   - 网络连接（下载模型和数据）
#
# 作者：zhangshengoo
# 版本：1.0.0
# =============================================================================

# 设置严格模式，遇到错误立即退出
set -e

# =============================================================================
# 默认配置参数
# =============================================================================
# 默认配置文件路径
CONFIG_FILE="config.yaml"

# 最大处理批次数（空表示无限制）
MAX_BATCHES=""

# 日志级别 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# 执行动作 (run, create-config)
ACTION="run"

# =============================================================================
# 使用说明函数
# =============================================================================
usage() {
    echo "ASR蒸馏框架启动脚本使用说明"
    echo "================================"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项说明:"
    echo "  -c, --config FILE      指定配置文件路径 (默认: config.yaml)"
    echo "  -b, --max-batches N    限制处理的批次数"
    echo "  -l, --log-level LEVEL  设置日志级别 (DEBUG|INFO|WARNING|ERROR)"
    echo "  -a, --action ACTION    指定执行动作 (run|create-config)"
    echo "  -o, --output FILE      输出文件路径 (用于create-config)"
    echo "  -h, --help             显示此帮助信息"
    echo ""
    echo "使用示例:"
    echo "  # 使用默认配置运行pipeline"
    echo "  $0"
    echo ""
    echo "  # 使用自定义配置文件"
    echo "  $0 -c my_config.yaml"
    echo ""
    echo "  # 限制处理100个批次，启用调试日志"
    echo "  $0 -b 100 -l DEBUG"
    echo ""
    echo "  # 创建新的配置文件"
    echo "  $0 -a create-config -o new_config.yaml"
    echo ""
    echo "  # 完整配置示例"
    echo "  $0 -c production.yaml -b 1000 -l INFO"
    echo ""
    echo "注意事项:"
    echo "  1. 请确保在项目根目录下运行此脚本"
    echo "  2. 首次运行前请先创建配置文件"
    echo "  3. GPU推理需要足够的显存（建议16GB+）"
    echo "  4. 大规模处理建议使用screen或tmux运行"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b|--max-batches)
            MAX_BATCHES="--max-batches $2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="--output $2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found. Please run this script from the project root directory."
    exit 1
fi

# Create necessary directories
mkdir -p logs data checkpoints

# Execute the requested action
case $ACTION in
    run)
        echo "Starting ASR Distillation Pipeline..."
        echo "Config: $CONFIG_FILE"
        echo "Log Level: $LOG_LEVEL"
        if [ -n "$MAX_BATCHES" ]; then
            echo "Max Batches: $MAX_BATCHES"
        fi
        echo ""
        
        python main.py run --config "$CONFIG_FILE" --log-level "$LOG_LEVEL" $MAX_BATCHES
        ;;
    create-config)
        echo "Creating configuration file..."
        python main.py create-config $OUTPUT_FILE
        ;;
    *)
        echo "Unknown action: $ACTION"
        usage
        exit 1
        ;;
esac