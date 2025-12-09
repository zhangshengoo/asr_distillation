"""
ConfigManager 简单单元测试

测试配置管理功能
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.config.manager import ConfigManager, ASRDistillationConfig


class TestConfigManager:
    """ConfigManager简单测试"""

    def test_init(self):
        """测试初始化"""
        manager = ConfigManager()
        assert manager.config_path is None
        assert manager.config is None

    def test_load_default_config(self):
        """测试加载默认配置"""
        manager = ConfigManager()
        config = manager.load_config()
        
        assert isinstance(config, ASRDistillationConfig)
        assert config.data.index_path == "./data/index"
        assert config.pipeline.num_cpu_workers == 10
        assert config.audio.target_sample_rate == 16000

    def test_validate_config(self):
        """测试配置验证"""
        manager = ConfigManager()
        config = manager.load_config()
        
        # 验证正常配置
        assert manager.validate_config(config) == True
        
        # 测试无效配置
        config.pipeline.num_cpu_workers = 0
        assert manager.validate_config(config) == False

    def test_update_config(self):
        """测试更新配置"""
        manager = ConfigManager()
        config = manager.load_config()
        
        # 更新配置
        updates = {
            'pipeline': {
                'num_cpu_workers': 20,
                'batch_size': 64
            }
        }
        
        manager.update_config(updates)
        updated_config = manager.get_config()
        
        assert updated_config.pipeline.num_cpu_workers == 20
        assert updated_config.pipeline.batch_size == 64

    def test_save_and_load_config(self):
        """测试保存和加载配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # 创建配置管理器
            manager = ConfigManager()
            config = manager.load_config()
            
            # 修改配置
            config.pipeline.num_cpu_workers = 15
            config.audio.target_sample_rate = 22050
            
            # 保存配置
            manager.save_config(config_path)
            
            # 验证文件存在
            assert Path(config_path).exists()
            
            # 加载配置
            new_manager = ConfigManager(config_path)
            loaded_config = new_manager.load_config()
            
            # 验证配置
            assert loaded_config.pipeline.num_cpu_workers == 15
            assert loaded_config.audio.target_sample_rate == 22050
            
        finally:
            # 清理临时文件
            Path(config_path).unlink(missing_ok=True)