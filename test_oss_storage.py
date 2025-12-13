#!/usr/bin/env python3
"""OSS Storage Test - Connectivity and Performance"""

import os
import sys
import time
import yaml
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.data.storage import OSSClient, MediaStorageManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_test_file(size_mb: float) -> Tuple[str, bytes]:
    """创建测试文件"""
    size_bytes = int(size_mb * 1024 * 1024)
    data = os.urandom(size_bytes)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
    temp_file.write(data)
    temp_file.close()
    
    return temp_file.name, data


def format_speed(bytes_per_second: float) -> str:
    """格式化速度"""
    mb_per_sec = bytes_per_second / (1024 * 1024)
    return f"{mb_per_sec:.2f} MB/s"


def test_connectivity(oss_client: OSSClient) -> bool:
    """测试OSS连通性"""
    try:
        # 使用OSSClient封装的方法进行连通性测试
        oss_client.list_objects(max_keys=1)
        return True
    except Exception as e:
        print(f"Connectivity failed: {e}")
        return False


def test_upload_speed(oss_client: OSSClient, test_prefix: str, size_mb: float = 10.0) -> float:
    """测试上传速度"""
    local_file, data = create_test_file(size_mb)
    test_key = f"{test_prefix}upload_test_{int(time.time())}.bin"
    
    try:
        start_time = time.time()
        oss_client.bucket.put_object_from_file(test_key, local_file)
        elapsed = time.time() - start_time
        
        speed = len(data) / elapsed
        
        # 清理
        oss_client.bucket.delete_object(test_key)
        os.unlink(local_file)
        
        return speed
    except Exception as e:
        print(f"Upload test failed: {e}")
        if os.path.exists(local_file):
            os.unlink(local_file)
        return 0.0


def test_download_speed(oss_client: OSSClient, test_prefix: str, size_mb: float = 10.0) -> float:
    """测试下载速度"""
    # 先上传测试文件
    local_file, data = create_test_file(size_mb)
    test_key = f"{test_prefix}download_test_{int(time.time())}.bin"
    
    try:
        oss_client.bucket.put_object_from_file(test_key, local_file)
        os.unlink(local_file)
        
        # 下载测试
        download_file = tempfile.NamedTemporaryFile(delete=False, suffix='.bin')
        download_file.close()
        
        start_time = time.time()
        oss_client.bucket.get_object_to_file(test_key, download_file.name)
        elapsed = time.time() - start_time
        
        speed = len(data) / elapsed
        
        # 清理
        oss_client.bucket.delete_object(test_key)
        os.unlink(download_file.name)
        
        return speed
    except Exception as e:
        print(f"Download test failed: {e}")
        try:
            oss_client.bucket.delete_object(test_key)
        except:
            pass
        return 0.0


def test_list_objects(storage_manager: MediaStorageManager) -> int:
    """测试文件列表功能"""
    try:
        files = storage_manager.list_media_files(prefix=storage_manager.audio_prefix)
        return len(files)
    except Exception as e:
        print(f"List objects failed: {e}")
        return -1


def run_tests(config_path: str, test_size_mb: float = 10.0):
    """运行所有测试"""
    print(f"\n{'='*60}")
    print(f"OSS Storage Test")
    print(f"{'='*60}\n")
    
    # 加载配置
    config = load_config(config_path)
    storage_config = config['data']['storage']
    
    print(f"Endpoint: {storage_config['endpoint']}")
    print(f"Bucket: {storage_config['bucket']}\n")
    
    # 初始化客户端
    try:
        oss_client = OSSClient(
            endpoint=storage_config['endpoint'],
            access_key_id=storage_config['access_key_id'],
            access_key_secret=storage_config['access_key_secret'],
            bucket_name=storage_config['bucket']
        )
        storage_manager = MediaStorageManager(storage_config)
    except Exception as e:
        print(f"Failed to initialize OSS client: {e}")
        return
    
    # 测试1: 连通性
    print("1. Connectivity Test")
    if test_connectivity(oss_client):
        print("   Status: OK\n")
    else:
        print("   Status: FAILED\n")
        return
    
    # 测试2: 上传速度
    print(f"2. Upload Speed Test ({test_size_mb} MB)")
    upload_speed = test_upload_speed(oss_client, storage_config.get('result_prefix', 'test/'), test_size_mb)
    if upload_speed > 0:
        print(f"   Speed: {format_speed(upload_speed)}\n")
    else:
        print("   Status: FAILED\n")
    
    # 测试3: 下载速度
    print(f"3. Download Speed Test ({test_size_mb} MB)")
    download_speed = test_download_speed(oss_client, storage_config.get('result_prefix', 'test/'), test_size_mb)
    if download_speed > 0:
        print(f"   Speed: {format_speed(download_speed)}\n")
    else:
        print("   Status: FAILED\n")
    
    # 测试4: 文件列表
    print("4. List Objects Test")
    num_files = test_list_objects(storage_manager)
    if num_files >= 0:
        print(f"   Found: {num_files} files\n")
    else:
        print("   Status: FAILED\n")
    
    # 总结
    print(f"{'='*60}")
    print("Test Summary:")
    print(f"  Upload Speed:   {format_speed(upload_speed) if upload_speed > 0 else 'N/A'}")
    print(f"  Download Speed: {format_speed(download_speed) if download_speed > 0 else 'N/A'}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Test OSS storage connectivity and performance')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    parser.add_argument(
        '--size',
        type=float,
        default=10.0,
        help='Test file size in MB (default: 10.0)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    run_tests(args.config, args.size)


if __name__ == '__main__':
    main()