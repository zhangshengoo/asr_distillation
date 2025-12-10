"""
Ray初始化测试

测试Ray集群初始化的各种场景，诊断ray.init()卡住的问题
"""

import pytest
import ray
import time
import psutil
import os
import tempfile
import signal
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config.manager import ConfigManager


class TestRayInitialization:
    """Ray初始化测试类"""
    
    def setup_method(self):
        """每个测试方法执行前的设置"""
        # 确保Ray未初始化
        if ray.is_initialized():
            ray.shutdown()
    
    def teardown_method(self):
        """每个测试方法执行后的清理"""
        # 确保Ray已关闭
        if ray.is_initialized():
            ray.shutdown()
    
    def test_basic_ray_init(self):
        """测试基本Ray初始化"""
        print("\n=== 测试基本Ray初始化 ===")
        
        start_time = time.time()
        try:
            ray.init()
            init_time = time.time() - start_time
            print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
            
            # 验证Ray状态
            assert ray.is_initialized(), "Ray应该已初始化"
            
            # 获取集群信息
            cluster_resources = ray.cluster_resources()
            print(f"集群资源: {cluster_resources}")
            
            # 获取节点信息
            nodes = ray.nodes()
            print(f"节点数量: {len(nodes)}")
            
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_ray_init_with_object_store_memory(self):
        """测试带对象存储内存配置的Ray初始化"""
        print("\n=== 测试带对象存储内存配置的Ray初始化 ===")
        
        # 测试不同的object_store_memory配置
        memory_configs = [
            100 * 1024 * 1024,    # 100MB
            500 * 1024 * 1024,    # 500MB
            1024 * 1024 * 1024,   # 1GB
        ]
        
        for memory_size in memory_configs:
            print(f"\n测试对象存储内存: {memory_size // (1024*1024)}MB")
            start_time = time.time()
            
            try:
                ray.init(object_store_memory=memory_size)
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 验证对象存储内存配置
                node_stats = ray.nodes()[0]["Resources"]
                print(f"节点资源: {node_stats}")
                
            except Exception as e:
                init_time = time.time() - start_time
                print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
                raise
            finally:
                if ray.is_initialized():
                    ray.shutdown()
    
    def test_ray_init_with_test_config(self):
        """测试使用测试配置文件的Ray初始化"""
        print("\n=== 测试使用测试配置文件的Ray初始化 ===")
        
        config_file = "tests/config_test.yaml"
        
        try:
            # 加载配置
            config_manager = ConfigManager(config_file)
            config = config_manager.load_config()
            
            print(f"测试配置文件加载成功，object_store_memory: {config.pipeline.object_store_memory}")
            print(f"CPU工作节点数: {config.pipeline.num_cpu_workers}")
            print(f"批处理大小: {config.pipeline.batch_size}")
            
            # 测试Ray初始化
            start_time = time.time()
            
            if not ray.is_initialized():
                ray.init(
                    object_store_memory=config.pipeline.object_store_memory,
                    ignore_reinit_error=True
                )
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 验证配置
                assert ray.is_initialized(), "Ray应该已初始化"
                
                # 获取集群信息
                cluster_resources = ray.cluster_resources()
                print(f"集群资源: {cluster_resources}")
            
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_ray_init_with_timeout(self):
        """测试带超时的Ray初始化"""
        print("\n=== 测试带超时的Ray初始化 ===")
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Ray初始化超时")
        
        # 设置30秒超时
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        start_time = time.time()
        try:
            ray.init()
            init_time = time.time() - start_time
            print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
            assert ray.is_initialized(), "Ray应该已初始化"
        except TimeoutError as e:
            init_time = time.time() - start_time
            print(f"Ray初始化超时，耗时: {init_time:.2f}秒")
            pytest.fail(f"Ray初始化超时: {e}")
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
            raise
        finally:
            signal.alarm(0)  # 取消超时
            signal.signal(signal.SIGALRM, original_handler)  # 恢复原始处理器
            if ray.is_initialized():
                ray.shutdown()
    
    def test_ray_init_environment_variables(self):
        """测试环境变量对Ray初始化的影响"""
        print("\n=== 测试环境变量对Ray初始化的影响 ===")
        
        # 保存原始环境变量
        original_env = os.environ.copy()
        
        # 测试不同的环境变量配置
        env_configs = [
            {
                "RAY_BACKEND_LOG_LEVEL": "debug",
                "RAY_LOG_TO_STDERR": "1"
            },
            {
                "RAY_object_store_memory": "1073741824",  # 1GB
                "RAY_BACKEND_LOG_LEVEL": "info"
            }
        ]
        
        for i, env_config in enumerate(env_configs):
            print(f"\n测试环境变量配置 {i+1}: {env_config}")
            
            # 设置环境变量
            for key, value in env_config.items():
                os.environ[key] = value
            
            try:
                start_time = time.time()
                ray.init(ignore_reinit_error=True)
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 获取Ray版本信息
                ray_version = ray.__version__
                print(f"Ray版本: {ray_version}")
                
            except Exception as e:
                init_time = time.time() - start_time
                print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
                raise
            finally:
                if ray.is_initialized():
                    ray.shutdown()
                # 恢复原始环境变量
                os.environ.clear()
                os.environ.update(original_env)
    
    def test_system_resources_impact(self):
        """测试系统资源对Ray初始化的影响"""
        print("\n=== 测试系统资源对Ray初始化的影响 ===")
        
        # 获取系统资源信息
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"CPU核心数: {cpu_count}")
        print(f"总内存: {memory.total / (1024**3):.2f}GB")
        print(f"可用内存: {memory.available / (1024**3):.2f}GB")
        print(f"磁盘总空间: {disk.total / (1024**3):.2f}GB")
        print(f"磁盘可用空间: {disk.free / (1024**3):.2f}GB")
        
        # 测试不同的资源配置
        resource_configs = [
            {"num_cpus": 1},
            {"num_cpus": min(cpu_count // 2, 4)},  # 限制最大CPU数
            {"num_cpus": 1, "memory": 1024 * 1024 * 1024},  # 1GB
        ]
        
        for i, resource_config in enumerate(resource_configs):
            print(f"\n测试资源配置 {i+1}: {resource_config}")
            
            try:
                start_time = time.time()
                ray.init(**resource_config)
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 验证资源配置
                cluster_resources = ray.cluster_resources()
                print(f"实际集群资源: {cluster_resources}")
                
            except Exception as e:
                init_time = time.time() - start_time
                print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
                # 不抛出异常，继续测试其他配置
                continue
            finally:
                if ray.is_initialized():
                    ray.shutdown()
    
    def test_ray_init_with_address(self):
        """测试指定地址的Ray初始化"""
        print("\n=== 测试指定地址的Ray初始化 ===")
        
        # 测试不同的地址配置
        addresses = [
            None,  # 本地模式
        ]
        
        for address in addresses:
            print(f"\n测试地址: {address}")
            
            try:
                start_time = time.time()
                
                if address:
                    ray.init(address=address, _redis_password="")
                else:
                    ray.init()
                    
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 获取节点信息
                nodes = ray.nodes()
                print(f"节点数量: {len(nodes)}")
                for node in nodes:
                    print(f"节点ID: {node['NodeID']}")
                    print(f"节点状态: {node['State']}")
                
            except Exception as e:
                init_time = time.time() - start_time
                print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
                # 某些地址可能不可用，继续测试其他配置
                continue
            finally:
                if ray.is_initialized():
                    ray.shutdown()
    
    def test_ray_init_logging(self):
        """测试Ray初始化日志输出"""
        print("\n=== 测试Ray初始化日志输出 ===")
        
        # 重定向日志输出
        import io
        import sys
        
        log_capture = io.StringIO()
        
        # 保存原始stdout
        original_stdout = sys.stdout
        
        try:
            # 重定向stdout
            sys.stdout = log_capture
            
            start_time = time.time()
            ray.init(log_to_driver=True)
            init_time = time.time() - start_time
            
            # 恢复stdout
            sys.stdout = original_stdout
            
            print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
            
            # 分析日志输出
            log_output = log_capture.getvalue()
            print(f"日志输出长度: {len(log_output)} 字符")
            
            if log_output:
                print("日志内容预览:")
                print(log_output[:500])  # 只显示前500个字符
            
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
            raise
        finally:
            # 恢复stdout
            sys.stdout = original_stdout
            if ray.is_initialized():
                ray.shutdown()
    
    def test_ray_init_with_test_config_mock(self):
        """使用模拟的测试配置测试Ray初始化"""
        print("\n=== 测试使用模拟测试配置的Ray初始化 ===")
        
        # 模拟配置对象
        class MockConfig:
            class Pipeline:
                object_store_memory = 134217728  # 128MB
                num_cpu_workers = 2
                num_gpu_workers = 0
                batch_size = 4
                max_concurrent_batches = 1
                checkpoint_interval = 10
                
                class cpu_worker_resources:
                    CPU = 1
                
                class gpu_worker_resources:
                    CPU = 1
                    GPU = 0
            
            pipeline = Pipeline()
        
        config = MockConfig()
        
        try:
            print(f"模拟配置加载成功，object_store_memory: {config.pipeline.object_store_memory}")
            
            # 测试Ray初始化
            start_time = time.time()
            
            if not ray.is_initialized():
                ray.init(
                    object_store_memory=config.pipeline.object_store_memory,
                    ignore_reinit_error=True
                )
                init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {init_time:.2f}秒")
                
                # 验证配置
                assert ray.is_initialized(), "Ray应该已初始化"
                
                # 获取集群信息
                cluster_resources = ray.cluster_resources()
                print(f"集群资源: {cluster_resources}")
            
        except Exception as e:
            init_time = time.time() - start_time
            print(f"Ray初始化失败，耗时: {init_time:.2f}秒，错误: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


class TestMonitoringSystemInteraction:
    """测试MonitoringSystem与Ray初始化交互的测试类"""
    
    def setup_method(self):
        """每个测试方法执行前的设置"""
        # 确保Ray未初始化
        if ray.is_initialized():
            ray.shutdown()
        
        # 清理Prometheus指标注册表，避免重复注册
        self._cleanup_prometheus_metrics()
    
    def teardown_method(self):
        """每个测试方法执行后的清理"""
        # 确保Ray已关闭
        if ray.is_initialized():
            ray.shutdown()
        
        # 清理Prometheus指标注册表
        self._cleanup_prometheus_metrics()
    
    def _cleanup_prometheus_metrics(self):
        """清理Prometheus指标注册表"""
        try:
            from prometheus_client import REGISTRY, CollectorRegistry
            # 创建新的注册表替换默认注册表
            REGISTRY._collector_to_names.clear()
            REGISTRY._names_to_collectors.clear()
            REGISTRY._collectors.clear()
        except Exception as e:
            print(f"清理Prometheus指标时出错: {e}")
            # 忽略清理错误，不影响测试
    
    def test_monitoring_system_before_ray_init(self):
        """测试在Ray初始化之前启动MonitoringSystem"""
        print("\n=== 测试在Ray初始化之前启动MonitoringSystem ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig
            
            # 创建测试配置
            config = MonitoringConfig(
                enable_prometheus=False,  # 测试环境不启用Prometheus
                prometheus_port=8001,
                metrics_interval=1.0,
                enable_gpu_monitoring=False,
                enable_ray_monitoring=False,
                checkpoint_interval=10,
                checkpoint_dir="./tests/test_data/checkpoints"
            )
            
            print("创建MonitoringSystem...")
            monitoring_system = MonitoringSystem(config)
            
            print("启动MonitoringSystem...")
            start_time = time.time()
            monitoring_system.start()
            monitoring_start_time = time.time() - start_time
            print(f"MonitoringSystem启动成功，耗时: {monitoring_start_time:.2f}秒")
            
            # 等待一下确保监控系统稳定
            time.sleep(2)
            
            print("初始化Ray...")
            start_time = time.time()
            ray.init()
            ray_init_time = time.time() - start_time
            print(f"Ray初始化成功，耗时: {ray_init_time:.2f}秒")
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
    
    def test_ray_init_before_monitoring_system(self):
        """测试在Ray初始化之后启动MonitoringSystem"""
        print("\n=== 测试在Ray初始化之后启动MonitoringSystem ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig
            
            print("初始化Ray...")
            start_time = time.time()
            ray.init()
            ray_init_time = time.time() - start_time
            print(f"Ray初始化成功，耗时: {ray_init_time:.2f}秒")
            
            # 创建测试配置
            config = MonitoringConfig(
                enable_prometheus=False,  # 测试环境不启用Prometheus
                prometheus_port=8001,
                metrics_interval=1.0,
                enable_gpu_monitoring=False,
                enable_ray_monitoring=True,  # 启用Ray监控
                checkpoint_interval=10,
                checkpoint_dir="./tests/test_data/checkpoints"
            )
            
            print("创建MonitoringSystem...")
            monitoring_system = MonitoringSystem(config)
            
            print("启动MonitoringSystem...")
            start_time = time.time()
            monitoring_system.start()
            monitoring_start_time = time.time() - start_time
            print(f"MonitoringSystem启动成功，耗时: {monitoring_start_time:.2f}秒")
            
            # 等待一下确保监控系统稳定
            time.sleep(2)
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_monitoring_system_with_ray_monitoring(self):
        """测试MonitoringSystem的Ray监控功能"""
        print("\n=== 测试MonitoringSystem的Ray监控功能 ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig
            
            print("初始化Ray...")
            ray.init()
            
            # 创建启用Ray监控的配置
            config = MonitoringConfig(
                enable_prometheus=False,
                prometheus_port=8001,
                metrics_interval=1.0,
                enable_gpu_monitoring=False,
                enable_ray_monitoring=True,  # 启用Ray监控
                checkpoint_interval=10,
                checkpoint_dir="./tests/test_data/checkpoints"
            )
            
            print("创建并启动MonitoringSystem...")
            monitoring_system = MonitoringSystem(config)
            monitoring_system.start()
            
            # 等待监控系统收集一些指标
            print("等待监控系统收集指标...")
            time.sleep(3)
            
            # 获取系统统计信息
            print("获取系统统计信息...")
            stats = monitoring_system.get_system_stats()
            print(f"系统统计: {stats}")
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_monitoring_system_with_prometheus(self):
        """测试MonitoringSystem的Prometheus功能"""
        print("\n=== 测试MonitoringSystem的Prometheus功能 ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig
            
            # 创建启用Prometheus的配置
            config = MonitoringConfig(
                enable_prometheus=True,  # 启用Prometheus
                prometheus_port=8001,    # 使用不同的端口
                metrics_interval=1.0,
                enable_gpu_monitoring=False,
                enable_ray_monitoring=False,
                checkpoint_interval=10,
                checkpoint_dir="./tests/test_data/checkpoints"
            )
            
            print("创建并启动MonitoringSystem（启用Prometheus）...")
            monitoring_system = MonitoringSystem(config)
            
            start_time = time.time()
            monitoring_system.start()
            monitoring_start_time = time.time() - start_time
            print(f"MonitoringSystem启动成功，耗时: {monitoring_start_time:.2f}秒")
            
            # 等待Prometheus服务器启动
            time.sleep(2)
            
            print("初始化Ray...")
            start_time = time.time()
            ray.init()
            ray_init_time = time.time() - start_time
            print(f"Ray初始化成功，耗时: {ray_init_time:.2f}秒")
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_monitoring_system_thread_safety(self):
        """测试MonitoringSystem的线程安全性"""
        print("\n=== 测试MonitoringSystem的线程安全性 ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig
            import threading
            
            print("初始化Ray...")
            ray.init()
            
            # 创建测试配置
            config = MonitoringConfig(
                enable_prometheus=False,
                prometheus_port=8001,
                metrics_interval=0.5,  # 较短的间隔
                enable_gpu_monitoring=False,
                enable_ray_monitoring=True,
                checkpoint_interval=10,
                checkpoint_dir="./tests/test_data/checkpoints"
            )
            
            print("创建MonitoringSystem...")
            monitoring_system = MonitoringSystem(config)
            
            # 启动监控系统
            monitoring_system.start()
            
            # 在多个线程中同时访问监控系统
            def worker_thread(thread_id):
                for i in range(5):
                    try:
                        stats = monitoring_system.get_system_stats()
                        print(f"线程 {thread_id} 获取统计信息: {len(stats)} 个字段")
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"线程 {thread_id} 出错: {e}")
            
            print("启动多个工作线程...")
            threads = []
            for i in range(3):
                t = threading.Thread(target=worker_thread, args=(i,))
                threads.append(t)
                t.start()
            
            # 等待所有线程完成
            for t in threads:
                t.join()
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_main_pipeline_scenario(self):
        """测试主要的pipeline场景，复现main.py中的问题"""
        print("\n=== 测试主要的pipeline场景 ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig, ConfigManager
            
            # 加载测试配置
            config_file = "tests/config_test.yaml"
            config_manager = ConfigManager(config_file)
            config = config_manager.load_config()
            
            print("加载测试配置成功")
            
            # 初始化监控系统（模拟main.py中的顺序）
            print("初始化监控系统...")
            monitoring_config = MonitoringConfig(**config.monitoring.__dict__)
            monitoring_system = MonitoringSystem(monitoring_config)
            
            start_time = time.time()
            monitoring_system.start()
            monitoring_start_time = time.time() - start_time
            print(f"监控系统启动成功，耗时: {monitoring_start_time:.2f}秒")
            
            # 等待监控系统稳定
            time.sleep(2)
            
            # 初始化Ray（模拟main.py中的顺序）
            print("初始化Ray...")
            if not ray.is_initialized():
                start_time = time.time()
                ray.init(
                    object_store_memory=config.pipeline.object_store_memory,
                    ignore_reinit_error=True
                )
                ray_init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {ray_init_time:.2f}秒")
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()
    
    def test_reverse_order_scenario(self):
        """测试反向顺序：先Ray后MonitoringSystem"""
        print("\n=== 测试反向顺序场景 ===")
        
        try:
            from src.monitoring.system import MonitoringSystem
            from src.config.manager import MonitoringConfig, ConfigManager
            
            # 加载测试配置
            config_file = "tests/config_test.yaml"
            config_manager = ConfigManager(config_file)
            config = config_manager.load_config()
            
            print("加载测试配置成功")
            
            # 先初始化Ray
            print("先初始化Ray...")
            if not ray.is_initialized():
                start_time = time.time()
                ray.init(
                    object_store_memory=config.pipeline.object_store_memory,
                    ignore_reinit_error=True
                )
                ray_init_time = time.time() - start_time
                print(f"Ray初始化成功，耗时: {ray_init_time:.2f}秒")
            
            # 再初始化监控系统
            print("再初始化监控系统...")
            monitoring_config = MonitoringConfig(**config.monitoring.__dict__)
            monitoring_system = MonitoringSystem(monitoring_config)
            
            start_time = time.time()
            monitoring_system.start()
            monitoring_start_time = time.time() - start_time
            print(f"监控系统启动成功，耗时: {monitoring_start_time:.2f}秒")
            
            # 停止监控系统
            monitoring_system.stop()
            
        except Exception as e:
            print(f"测试失败: {e}")
            raise
        finally:
            if ray.is_initialized():
                ray.shutdown()


def run_diagnostic_tests():
    """运行诊断测试的主函数"""
    print("开始Ray初始化诊断测试...")
    print(f"Python版本: {psutil.sys.version}")
    print(f"Ray版本: {ray.__version__}")
    print(f"系统平台: {psutil.sys.platform}")
    print(f"CPU架构: {psutil.sys.architecture}")
    
    # 运行基础Ray初始化测试
    print("\n" + "="*60)
    print("基础Ray初始化测试")
    print("="*60)
    
    test_instance = TestRayInitialization()
    
    basic_tests = [
        test_instance.test_basic_ray_init,
        test_instance.test_ray_init_with_object_store_memory,
        test_instance.test_ray_init_with_test_config,
        test_instance.test_ray_init_with_timeout,
    ]
    
    for test in basic_tests:
        try:
            test()
        except Exception as e:
            print(f"测试失败: {test.__name__}, 错误: {e}")
        print("-" * 50)
    
    # 运行MonitoringSystem交互测试
    print("\n" + "="*60)
    print("MonitoringSystem交互测试")
    print("="*60)
    
    monitoring_test_instance = TestMonitoringSystemInteraction()
    
    monitoring_tests = [
        monitoring_test_instance.test_monitoring_system_before_ray_init,
        monitoring_test_instance.test_ray_init_before_monitoring_system,
        monitoring_test_instance.test_monitoring_system_with_ray_monitoring,
        monitoring_test_instance.test_monitoring_system_with_prometheus,
        monitoring_test_instance.test_main_pipeline_scenario,
        monitoring_test_instance.test_reverse_order_scenario,
    ]
    
    for test in monitoring_tests:
        try:
            test()
        except Exception as e:
            print(f"测试失败: {test.__name__}, 错误: {e}")
        print("-" * 50)
    
    print("Ray初始化诊断测试完成")


if __name__ == "__main__":
    run_diagnostic_tests()
