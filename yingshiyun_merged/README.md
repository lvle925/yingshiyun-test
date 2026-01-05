# 萤石云统一服务

## 项目简介

本项目是将9个独立的萤石云相关项目合并为一个标准的Router-Service-Schema分层架构的统一服务。

## 项目结构

```
yingshiyun_merged/
├── routers/              # 路由层 - 处理HTTP请求和响应
├── services/             # 服务层 - 业务逻辑处理
│   ├── leinuo/          # 雷诺相关服务
│   ├── qimen/           # 奇门相关服务
│   ├── ziwei/           # 紫微相关服务
│   ├── summary/         # 总结服务
│   ├── year_score/      # 年运势评分服务
│   ├── llm/             # LLM通用服务
│   ├── session/         # 会话管理服务
│   └── validation/      # 验证服务
├── schemas/             # 数据模型层 - Pydantic模型定义
├── database/            # 数据库层
├── utils/               # 工具函数层
├── clients/             # 外部客户端层
├── prompts/             # 提示词配置
├── security/            # 安全层
├── monitoring/          # 监控层
├── assets/              # 静态资源
├── logs/                # 日志目录
├── config.py            # 全局配置
├── main.py              # 应用入口
├── requirements.txt     # 依赖管理
├── .env.example         # 环境变量模板
├── Dockerfile           # Docker配置
└── README.md            # 项目说明
```

## 原项目映射关系

| 原项目 | 新位置 | 说明 |
|--------|--------|------|
| 10_10_leinuo_yunshi_day | services/leinuo/day_service.py | 雷诺每日运势 |
| 11_5_yingshi_yunshi_year_score_ali | services/year_score/score_service.py | 年运势评分 |
| 12_11_yingshi_yunshi_day_new_qimen | services/qimen/day_service.py | 奇门每日运势 |
| 12_12_llm_yingshi_leinuo_ali | services/leinuo/llm_service.py | LLM雷诺 |
| 12_12_llm_yingshi_qimen_ali | services/qimen/llm_service.py | LLM奇门 |
| 12_12_llm_yingshi_summary_ali | services/summary/summary_service.py | LLM总结 |
| 12_12_llm_yingshi_ziwei_ali | services/ziwei/llm_service.py | LLM紫微 |
| report_prediction_bendi_test_12_8 | services/ziwei/report_service.py | 紫微报告 |
| ziwei_report_year_aliyun_11_12 | services/ziwei/year_service.py | 紫微年度报告 |

## 安装和运行

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入实际配置
vim .env
```

### 3. 运行服务

```bash
# 开发模式
python main.py

# 或使用uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Docker部署

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

## API文档

服务启动后，访问以下地址查看API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 架构特点

### 1. 分层架构

- **Router层**: 负责HTTP请求处理、参数验证、响应格式化
- **Service层**: 负责业务逻辑实现、数据处理、外部调用
- **Schema层**: 负责数据模型定义、请求/响应验证

### 2. 模块化设计

- 每个业务模块独立管理
- 共享模块统一维护
- 易于扩展和维护

### 3. 配置统一

- 所有配置集中在config.py
- 环境变量统一管理
- 支持多环境配置

### 4. 代码复用

- 共享工具函数统一放在utils/
- 数据库操作统一放在database/
- 外部客户端统一放在clients/

## 开发指南

### 添加新的API端点

1. 在`routers/`目录下创建或编辑路由文件
2. 在`services/`对应模块下实现业务逻辑
3. 在`schemas/`下定义请求/响应模型
4. 在`main.py`中注册路由

### 添加新的工具函数

1. 在`utils/`目录下创建或编辑工具文件
2. 在需要使用的地方导入

### 添加新的数据库操作

1. 在`database/`目录下编辑数据库管理文件
2. 在service层调用数据库操作

## 注意事项

1. **保留原有接口**: 所有原项目的API端点路径和名称保持不变
2. **保留业务逻辑**: 所有业务逻辑代码完整迁移，不做修改
3. **更新导入路径**: 所有import语句需要更新为新的模块路径
4. **配置文件**: 确保.env文件配置正确
5. **依赖安装**: 确保所有依赖正确安装

## 待完成任务

- [ ] 创建所有路由文件并实现路由逻辑
- [ ] 更新所有服务文件中的import路径
- [ ] 在main.py中注册所有路由
- [ ] 测试所有API端点
- [ ] 编写单元测试
- [ ] 完善API文档
