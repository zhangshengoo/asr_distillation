# ASR蒸馏框架测试模块

## 简单说明

这是一个简单易懂的单元测试模块，用于测试ASR蒸馏框架的各个组件。

## 测试结构

```
tests/
├── __init__.py              # 测试模块初始化
├── conftest.py              # pytest配置和共享fixtures
├── run_tests.py             # 测试运行脚本
├── README.md                # 说明文档
├── compute/                 # 计算层测试
│   ├── test_batch_inference.py    # BatchInferenceStage测试
│   └── test_audio_processor.py     # 音频处理测试
├── data/                    # 数据层测试
│   └── test_audio_indexer.py       # 音频索引测试
└── config/                  # 配置管理测试
    └── test_manager.py             # 配置管理器测试
```

## 快速开始

### 1. 安装测试依赖

```bash
pip install pytest pytest-asyncio pytest-cov
```

### 2. 运行测试

#### 方法一：使用测试运行脚本
```bash
cd tests/
python run_tests.py
```

#### 方法二：直接使用pytest
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/compute/test_batch_inference.py -v

# 运行特定测试方法
pytest tests/compute/test_batch_inference.py::TestBatchInferenceStage::test_init -v
```

## 测试覆盖的功能

### BatchInferenceStage测试
- ✅ 初始化配置
- ✅ 正常批次处理
- ✅ 错误处理
- ✅ 空批次处理

### AudioProcessor测试
- ✅ 音频预处理配置
- ✅ 音频归一化
- ✅ 音频截断和填充
- ✅ 音频重采样

### AudioIndexer测试
- ✅ 索引创建
- ✅ 空索引加载
- ✅ 音频元数据

### ConfigManager测试
- ✅ 默认配置加载
- ✅ 配置验证
- ✅ 配置更新
- ✅ 配置保存和加载

## 测试最佳实践

1. **保持简单**：每个测试只验证一个功能点
2. **使用Mock**：避免依赖外部资源
3. **清晰命名**：测试方法名要清楚表达测试内容
4. **独立运行**：每个测试都应该能独立运行

## 添加新测试

1. 在对应目录创建测试文件
2. 继承简单的测试模式
3. 使用Mock避免外部依赖
4. 运行测试确保通过

## 故障排除

### 常见问题

1. **导入错误**：确保在项目根目录运行测试
2. **Mock失败**：检查Mock对象的方法名和参数
3. **断言失败**：验证测试数据和预期结果

### 调试技巧

```bash
# 显示详细输出
pytest tests/ -v -s

# 只运行失败的测试
pytest tests/ --lf

# 在测试中添加print语句调试
```

## 贡献指南

欢迎添加更多测试！请遵循以下原则：

1. 保持测试简单易懂
2. 一个测试一个功能点
3. 使用有意义的测试数据
4. 添加必要的注释说明