#!/bin/bash

# ASR Distillation Framework Startup Script

set -e

# Default values
CONFIG_FILE="config.yaml"
MAX_BATCHES=""
LOG_LEVEL="INFO"
ACTION="run"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --config FILE      Configuration file (default: config.yaml)"
    echo "  -b, --max-batches N    Maximum number of batches to process"
    echo "  -l, --log-level LEVEL  Log level (DEBUG, INFO, WARNING, ERROR)"
    echo "  -a, --action ACTION    Action to perform (run, create-config)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run with default config"
    echo "  $0 -c my_config.yaml                  # Run with custom config"
    echo "  $0 -b 100 -l DEBUG                   # Run with 100 batches and debug logging"
    echo "  $0 -a create-config -o new_config.yaml  # Create new config file"
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