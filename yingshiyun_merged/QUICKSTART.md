# 快速开始指南

## 项目说明

本项目是将9个独立的萤石云相关项目合并为一个标准的Router-Service-Schema分层架构的统一服务。

**当前状态**: 基础架构已完成，所有业务代码已迁移到位，但需要完成路由分离和导入路径更新才能运行。

## 项目结构

```
yingshiyun_merged/
├── routers/              # 路由层（需要实现）
├── services/             # 服务层（已完成文件迁移）
├── schemas/              # 数据模型层（需要完善）
├── database/             # 数据库层（已完成）
├── utils/                # 工具函数层（已完成）
├── clients/              # 外部客户端层（已完成）
├── prompts/              # 提示词配置（已完成）
├── security/             # 安全层（已完成）
├── monitoring/           # 监控层（已完成）
├── assets/               # 静态资源（已完成）
└── main.py               # 应用入口（框架已完成）
```

## 文件统计

- **Python文件**: 72个
- **配置文件**: 17个
- **总文件数**: 95个
- **目录数**: 23个

## 已完成的工作

### ✅ 基础架构
- 完整的分层目录结构
- 所有__init__.py文件
- 统一的配置管理（config.py）
- 环境变量模板（.env.example）
- Docker配置（Dockerfile）

### ✅ 业务代码迁移
所有9个项目的业务逻辑代码已完整复制到services/目录：

| 原项目 | 新位置 |
|--------|--------|
| 10_10_leinuo_yunshi_day | services/leinuo/day_service.py |
| 11_5_yingshi_yunshi_year_score_ali | services/year_score/score_service.py |
| 12_11_yingshi_yunshi_day_new_qimen | services/qimen/day_service.py |
| 12_12_llm_yingshi_leinuo_ali | services/leinuo/llm_service.py |
| 12_12_llm_yingshi_qimen_ali | services/qimen/llm_service.py |
| 12_12_llm_yingshi_summary_ali | services/summary/summary_service.py |
| 12_12_llm_yingshi_ziwei_ali | services/ziwei/llm_service.py |
| report_prediction_bendi_test_12_8 | services/ziwei/report_service.py |
| ziwei_report_year_aliyun_11_12 | services/ziwei/year_service.py |

### ✅ 共享模块
- utils/ - 所有工具函数（8个文件）
- database/ - 数据库管理（3个文件）
- clients/ - 外部客户端（4个文件）
- prompts/ - 提示词配置（多个文件）
- security/ - 安全验证
- monitoring/ - 监控工具

### ✅ 文档
- README.md - 项目说明
- MIGRATION_GUIDE.md - 详细迁移指南
- QUICKSTART.md - 本文档
- PROJECT_MERGE_REPORT.md - 合并报告

## 待完成的工作

### ⚠️ 必须完成才能运行

#### 1. 创建路由文件
需要为每个服务创建路由文件，将API端点从服务层分离：

```bash
# 需要创建的路由文件
routers/leinuo_day.py      # 雷诺每日运势路由
routers/leinuo_llm.py      # LLM雷诺路由
routers/qimen_day.py       # 奇门每日运势路由
routers/qimen_llm.py       # LLM奇门路由
routers/ziwei_llm.py       # LLM紫微路由
routers/ziwei_report.py    # 紫微报告路由
routers/ziwei_year.py      # 紫微年度报告路由
routers/year_score.py      # 年运势评分路由
routers/summary.py         # 总结路由
```

#### 2. 更新导入路径
所有服务文件中的导入需要更新：

**示例更新**:
```python
# 原来
import config
from db_manager2 import *
from liushifunction import *

# 改为
from config import VLLM_API_BASE_URL, DB_CONFIG
from database.db_manager2 import *
from utils.liushifunction import *
```

#### 3. 创建Schema文件
为每个服务创建Pydantic模型：

```bash
schemas/leinuo.py      # 雷诺相关模型
schemas/qimen.py       # 奇门相关模型
schemas/summary.py     # 总结相关模型
schemas/year_score.py  # 年运势评分模型
schemas/common.py      # 通用模型
```

#### 4. 注册路由
在main.py中取消注释并注册所有路由。

## 使用步骤

### 步骤1: 解压项目

```bash
tar -xzf yingshiyun_merged.tar.gz
cd yingshiyun_merged
```

### 步骤2: 配置环境

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入实际配置
vim .env
```

### 步骤3: 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤4: 完成代码迁移

按照 MIGRATION_GUIDE.md 完成以下工作：

1. 创建路由文件
2. 更新导入路径
3. 创建Schema文件
4. 注册路由

### 步骤5: 运行服务

```bash
# 开发模式
python main.py

# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 步骤6: 访问API文档

服务启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- 健康检查: http://localhost:8000/health

## Docker部署

```bash
# 构建镜像
docker build -t yingshiyun:latest .

# 运行容器
docker run -d \
  --name yingshiyun \
  -p 8000:8000 \
  --env-file .env \
  yingshiyun:latest
```

## 重要提示

### ⚠️ 当前状态
项目已完成基础架构搭建和文件迁移，但**不能直接运行**。需要完成以下工作：

1. **路由分离**: 将API端点从服务层分离到路由层
2. **导入更新**: 更新所有文件的导入路径
3. **Schema创建**: 创建完整的数据模型定义

### ✅ 保证事项
- 所有业务逻辑代码已完整保留
- 所有文件引用已复制到位
- 所有接口名称将保持不变
- 配置已统一管理

### 📚 参考文档
- **MIGRATION_GUIDE.md**: 详细的迁移步骤和示例
- **README.md**: 完整的项目说明
- **PROJECT_MERGE_REPORT.md**: 合并工作报告

## 目录说明

### routers/ - 路由层
负责HTTP请求处理、参数验证、响应格式化。每个路由文件对应一个业务模块。

### services/ - 服务层
负责业务逻辑实现、数据处理、外部调用。已包含所有9个项目的业务代码。

### schemas/ - 数据模型层
负责数据模型定义、请求/响应验证。使用Pydantic定义所有数据结构。

### database/ - 数据库层
统一的数据库连接和操作管理。包含3个数据库管理器。

### utils/ - 工具函数层
共享的工具函数，包括紫微、奇门、天干地支等算法。

### clients/ - 外部客户端层
外部API和服务的客户端封装，包括VLLM、占星API等。

### prompts/ - 提示词配置
LLM提示词配置文件，按业务模块组织。

### security/ - 安全层
签名验证、权限控制等安全相关功能。

### monitoring/ - 监控层
服务监控、性能指标收集等。

### assets/ - 静态资源
CSV、XLSX等数据文件。

## 技术栈

- **Web框架**: FastAPI 0.116.0
- **异步支持**: aiohttp, aiomysql
- **数据库**: MySQL
- **LLM**: OpenAI, LangChain
- **数据处理**: Pandas, NumPy
- **日期处理**: chinese_calendar, lunardate, zhdate
- **监控**: Prometheus, psutil
- **缓存**: Redis

## 下一步建议

1. **优先完成路由层**: 先实现一个模块（如雷诺每日运势），作为模板
2. **逐步迁移**: 按模块逐个完成迁移和测试
3. **编写测试**: 为每个API端点编写测试用例
4. **完善文档**: 补充API文档和使用说明

## 获取帮助

- 查看 MIGRATION_GUIDE.md 了解详细的迁移步骤
- 查看 README.md 了解项目整体架构
- 查看 PROJECT_MERGE_REPORT.md 了解合并详情

