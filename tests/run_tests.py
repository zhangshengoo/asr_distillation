#!/usr/bin/env python3
"""
ASR蒸馏框架测试运行脚本

简单易用的测试运行工具
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*50}")
    print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("警告:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print("输出:", e.stdout)
        print("错误:", e.stderr)
        return False


def main():
    """主函数"""
    print("ASR蒸馏框架测试运行器")
    print("=" * 50)
    
    # 检查pytest是否安装
    try:
        import pytest
        print(f"pytest版本: {pytest.__version__}")
    except ImportError:
        print("错误: pytest未安装，请运行: pip install pytest")
        sys.exit(1)
    
    # 测试选项
    test_options = {
        '1': ("运行所有测试", ["python", "-m", "pytest", "tests/", "-v"]),
        '2': ("只运行单元测试", ["python", "-m", "pytest", "tests/", "-v", "-m", "unit"]),
        '3': ("运行BatchInferenceStage测试", ["python", "-m", "pytest", "tests/compute/test_batch_inference.py", "-v"]),
        '4': ("运行音频处理测试", ["python", "-m", "pytest", "tests/compute/test_audio_processor.py", "-v"]),
        '5': ("运行配置管理测试", ["python", "-m", "pytest", "tests/config/test_manager.py", "-v"]),
        '6': ("运行数据索引测试", ["python", "-m", "pytest", "tests/data/test_audio_indexer.py", "-v"]),
        '7': ("显示测试覆盖率", ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing"]),
        '8': ("运行特定测试文件", None),  # 需要用户输入
        '0': ("退出", None)
    }
    
    while True:
        print("\n请选择测试选项:")
        for key, (desc, _) in test_options.items():
            print(f"  {key}. {desc}")
        
        choice = input("\n请输入选项 (0-8): ").strip()
        
        if choice == '0':
            print("退出测试运行器")
            break
        elif choice in test_options:
            desc, cmd = test_options[choice]
            
            if choice == '8':
                # 用户输入特定测试文件
                test_file = input("请输入测试文件路径 (如: tests/compute/test_batch_inference.py): ").strip()
                if test_file:
                    cmd = ["python", "-m", "pytest", test_file, "-v"]
                else:
                    print("无效的文件路径")
                    continue
            
            if cmd:
                success = run_command(cmd, desc)
                if success:
                    print(f"\n✅ {desc} 完成")
                else:
                    print(f"\n❌ {desc} 失败")
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    main()