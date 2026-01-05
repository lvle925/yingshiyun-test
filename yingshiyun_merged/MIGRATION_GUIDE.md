# 项目迁移指南

## 概述

本文档说明如何完成从9个独立项目到统一Router-Service-Schema架构的迁移。

## 已完成的工作

### 1. 目录结构创建 ✅

已创建完整的分层目录结构：
- routers/ - 路由层
- services/ - 服务层（包含所有子模块）
- schemas/ - 数据模型层
- database/ - 数据库层
- utils/ - 工具函数层
- clients/ - 外部客户端层
- prompts/ - 提示词配置
- security/ - 安全层
- monitoring/ - 监控层
- assets/ - 静态资源

### 2. 文件复制 ✅

已完成所有业务逻辑文件的复制：

#### 工具函数 (utils/)
- liushifunction.py - 紫微六十四卦函数
- tiangan_function.py - 天干地支函数
- xingyaoxingzhi.py - 星曜星质
- ziwei_ai_function.py - 紫微AI函数
- tokenizer.py - 分词工具
- utils.py - 通用工具
- processing.py - 数据处理
- time_filters.py - 时间过滤

#### 数据库 (database/)
- db_manager.py - 主数据库管理器
- db_manager2.py - 备用数据库管理器
- db_manager_yingshi.py - 影视专用数据库管理器

#### 客户端 (clients/)
- vllm_client.py - VLLM客户端
- external_api_client.py - 外部API客户端
- astro_api_client.py - 占星API客户端
- shared_client.py - 共享客户端

#### 服务层 (services/)
- leinuo/leinuo_day_service.py - 雷诺每日运势服务
- leinuo/leinuo_llm_service.py - LLM雷诺服务
- qimen/qimen_day_service.py - 奇门每日运势服务
- qimen/qimen_leinuo_llm_service.py - LLM奇门服务
- qimen/api_main_*.py - 奇门多个API入口
- ziwei/ziwei_leinuo_llm_service.py - LLM紫微服务
- ziwei/ziwei_report_service.py - 紫微报告服务
- ziwei/ziwei_year_service.py - 紫微年度报告服务
- summary/summary_service.py - 总结服务
- year_score/year_score_service.py - 年运势评分服务
- llm/llm_calls.py - LLM调用
- llm/llm_response.py - LLM响应
- session/session_manager.py - 会话管理
- validation/query_intent.py - 意图识别
- validation/validation_rules.py - 验证规则

#### 其他模块
- security/verifier.py - 签名验证
- monitoring/monitor.py - 监控工具
- prompts/ - 所有提示词配置文件
- schemas/ziwei.py - 紫微数据模型
- assets/ - 所有静态资源文件

### 3. 配置文件 ✅

- config.py - 统一配置文件
- .env.example - 环境变量模板
- requirements.txt - 依赖管理
- Dockerfile - Docker配置
- .gitignore - Git忽略文件

### 4. 主入口文件 ✅

- main.py - FastAPI应用入口

## 待完成的工作

### 1. 创建路由文件 ⚠️

需要为每个服务创建对应的路由文件，将API端点从服务文件中分离出来：

#### routers/leinuo_day.py
从 `services/leinuo/leinuo_day_service.py` 中提取路由：
```python
from fastapi import APIRouter
from services.leinuo import day_service

router = APIRouter()

# 将原来的 @app.post() 改为 @router.post()
# 业务逻辑调用 day_service 中的函数
```

#### routers/leinuo_llm.py
从 `services/leinuo/leinuo_llm_service.py` 中提取路由

#### routers/qimen_day.py
从 `services/qimen/api_main_*.py` 中提取路由

#### routers/qimen_llm.py
从 `services/qimen/qimen_llm_service.py` 中提取路由

#### routers/ziwei_llm.py
从 `services/ziwei/ziwei_llm_service.py` 中提取路由

#### routers/ziwei_report.py
从 `services/ziwei/ziwei_report_service.py` 中提取路由

#### routers/ziwei_year.py
从 `services/ziwei/ziwei_year_service.py` 中提取路由

#### routers/year_score.py
从 `services/year_score/year_score_service.py` 中提取路由

#### routers/summary.py
从 `services/summary/summary_service.py` 中提取路由

### 2. 更新导入路径 ⚠️

所有服务文件中的导入路径需要更新：

#### 示例：更新 services/leinuo/leinuo_day_service.py

原来的导入：
```python
import pandas as pd
```

需要更新为：
```python
from utils import processing
from database import db_manager
from clients import vllm_client
from config import VLLM_API_BASE_URL, DB_CONFIG
```

#### 需要更新的文件列表

**雷诺模块**
- [ ] services/leinuo/leinuo_day_service.py
- [ ] services/leinuo/leinuo_llm_service.py

**奇门模块**
- [ ] services/qimen/qimen_day_service.py
- [ ] services/qimen/qimen_llm_service.py
- [ ] services/qimen/api_main_day.py
- [ ] services/qimen/api_main_calendar.py
- [ ] services/qimen/api_main_attributes.py
- [ ] services/qimen/db_query.py
- [ ] services/qimen/user_info_extractor.py

**紫微模块**
- [ ] services/ziwei/ziwei_llm_service.py
- [ ] services/ziwei/ziwei_report_service.py
- [ ] services/ziwei/ziwei_year_service.py
- [ ] services/ziwei/chat_processor.py
- [ ] services/ziwei/ziwei_analyzer.py
- [ ] services/ziwei/chat_time_utils.py

**其他模块**
- [ ] services/summary/summary_service.py
- [ ] services/year_score/year_score_service.py
- [ ] services/llm/llm_calls.py
- [ ] services/llm/llm_response.py
- [ ] services/session/session_manager.py
- [ ] services/validation/query_intent.py
- [ ] services/validation/validation_rules.py

**工具模块**
- [ ] utils/liushifunction.py
- [ ] utils/tiangan_function.py
- [ ] utils/xingyaoxingzhi.py
- [ ] utils/ziwei_ai_function.py
- [ ] utils/processing.py
- [ ] utils/time_filters.py

**数据库模块**
- [ ] database/db_manager.py
- [ ] database/db_manager2.py
- [ ] database/db_manager_yingshi.py

**客户端模块**
- [ ] clients/vllm_client.py
- [ ] clients/external_api_client.py
- [ ] clients/astro_api_client.py
- [ ] clients/shared_client.py

**其他模块**
- [ ] security/verifier.py
- [ ] monitoring/monitor.py
- [ ] monitoring/monitor_qimen.py
- [ ] prompts/prompt_logic.py
- [ ] prompts/prompt_manager.py

### 3. 创建Schema文件 ⚠️

需要为每个服务创建对应的Pydantic模型：

- [ ] schemas/leinuo.py - 雷诺相关模型
- [ ] schemas/qimen.py - 奇门相关模型
- [ ] schemas/summary.py - 总结相关模型
- [ ] schemas/year_score.py - 年运势评分模型
- [ ] schemas/common.py - 通用模型

### 4. 注册路由 ⚠️

在 main.py 中取消注释并注册所有路由：

```python
from routers import leinuo_day, leinuo_llm
from routers import qimen_day, qimen_llm
from routers import ziwei_llm, ziwei_report, ziwei_year
from routers import year_score, summary

app.include_router(leinuo_day.router, prefix="/leinuo/day", tags=["雷诺每日运势"])
app.include_router(leinuo_llm.router, prefix="/leinuo/llm", tags=["LLM雷诺"])
# ... 其他路由
```

### 5. 更新配置引用 ⚠️

所有服务文件中的配置读取需要统一：

原来：
```python
VLLM_API_BASE_URL = os.getenv("VLLM_API_BASE_URL")
```

改为：
```python
from config import VLLM_API_BASE_URL
```

### 6. 测试验证 ⚠️

- [ ] 测试每个API端点
- [ ] 验证数据库连接
- [ ] 验证外部API调用
- [ ] 验证会话管理
- [ ] 验证签名验证
- [ ] 性能测试

## 导入路径更新规则

### 通用规则

| 原导入 | 新导入 |
|--------|--------|
| `from config import *` | `from config import VLLM_API_BASE_URL, DB_CONFIG, ...` |
| `import db_manager2` | `from database import db_manager2` |
| `from liushifunction import *` | `from utils.liushifunction import *` |
| `from tiangan_function import *` | `from utils.tiangan_function import *` |
| `from utils import *` | `from utils.utils import *` |
| `from tokenizer import *` | `from utils.tokenizer import *` |
| `from session_manager import *` | `from services.session.session_manager import *` |
| `from validation_rules import *` | `from services.validation.validation_rules import *` |
| `from queryIntent import *` | `from services.validation.query_intent import *` |

### 特定模块规则

#### 奇门模块
```python
# 原来
from helper_libs.db_manager_yingshi import *
from app.llm_calls import *
from app.prompt_manager import *

# 改为
from database.db_manager_yingshi import *
from services.llm.llm_calls import *
from prompts.prompt_manager import *
```

#### 紫微模块
```python
# 原来
from database.db_manager import *
from clients.vllm_client import *
from services.chat_processor import *
from security.verifier import *

# 改为
from database.db_manager import *
from clients.vllm_client import *
from services.ziwei.chat_processor import *
from security.verifier import *
```

## 路由分离示例

### 原代码结构（services/leinuo/leinuo_day_service.py）

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/divination")
async def divination(request: ClientRequest):
    # 业务逻辑
    pass
```

### 分离后结构

#### routers/leinuo_day.py（路由层）
```python
from fastapi import APIRouter
from schemas.leinuo import DivinationRequest, DivinationResponse
from services.leinuo import day_service

router = APIRouter()

@router.post("/api/divination", response_model=DivinationResponse)
async def divination(request: DivinationRequest):
    return await day_service.process_divination(request)
```

#### services/leinuo/leinuo_day_service.py（服务层）
```python
from schemas.leinuo import DivinationRequest, DivinationResponse
from database import db_manager
from utils import processing
from config import VLLM_API_BASE_URL

async def process_divination(request: DivinationRequest) -> DivinationResponse:
    # 业务逻辑实现
    pass
```

#### schemas/leinuo.py（模型层）
```python
from pydantic import BaseModel

class DivinationRequest(BaseModel):
    appid: str
    prompt: str
    # ... 其他字段

class DivinationResponse(BaseModel):
    status: str
    result: dict
    # ... 其他字段
```

## 注意事项

1. **保持接口不变**: 所有API端点的路径、参数、响应格式必须与原项目完全一致
2. **保留业务逻辑**: 不要修改任何业务逻辑代码，只做路径调整
3. **逐步迁移**: 建议一个模块一个模块地迁移和测试
4. **备份原文件**: 在修改前确保原文件已备份
5. **测试驱动**: 每完成一个模块的迁移，立即进行测试

## 迁移优先级

建议按以下顺序进行迁移：

1. **第一批**: 工具模块和数据库模块（无依赖）
   - utils/
   - database/
   - clients/

2. **第二批**: 共享服务模块
   - services/session/
   - services/validation/
   - services/llm/

3. **第三批**: 业务服务模块
   - services/leinuo/
   - services/qimen/
   - services/ziwei/
   - services/summary/
   - services/year_score/

4. **第四批**: 路由层
   - routers/

5. **第五批**: 主入口
   - main.py

## 验证清单

完成迁移后，使用以下清单验证：

- [ ] 所有服务文件的导入路径已更新
- [ ] 所有路由文件已创建并正确引用服务
- [ ] 所有Schema文件已创建
- [ ] main.py中所有路由已注册
- [ ] 配置文件正确加载
- [ ] 数据库连接正常
- [ ] 所有API端点可以正常访问
- [ ] 业务逻辑功能正常
- [ ] 日志输出正常
- [ ] 错误处理正常

## 常见问题

### Q: 如何处理循环导入？
A: 使用类型提示的字符串形式，或将导入移到函数内部

### Q: 如何处理相对导入？
A: 统一使用绝对导入，从项目根目录开始

### Q: 如何处理配置文件中的路径？
A: 使用相对于项目根目录的路径，或使用绝对路径

### Q: 如何测试单个模块？
A: 创建单元测试文件，使用pytest进行测试

## 总结

本迁移指南提供了完整的步骤和示例，帮助将9个独立项目合并为统一的Router-Service-Schema架构。关键是保持接口不变，只调整内部结构，确保业务逻辑完整迁移。
