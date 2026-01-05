# Router-Service-Schema 分层架构设计

## 架构概述

将9个独立项目合并为统一的分层架构：

```
yingshiyun/
├── routers/              # 路由层 - 处理HTTP请求和响应
│   ├── __init__.py
│   ├── leinuo_day.py     # 雷诺每日运势路由
│   ├── leinuo_llm.py     # LLM雷诺路由
│   ├── qimen_day.py      # 奇门每日运势路由
│   ├── qimen_llm.py      # LLM奇门路由
│   ├── ziwei_llm.py      # LLM紫微路由
│   ├── ziwei_report.py   # 紫微报告路由
│   ├── ziwei_year.py     # 紫微年度报告路由
│   ├── year_score.py     # 年运势评分路由
│   └── summary.py        # 总结路由
│
├── services/             # 服务层 - 业务逻辑处理
│   ├── __init__.py
│   ├── leinuo/           # 雷诺相关服务
│   │   ├── __init__.py
│   │   ├── day_service.py
│   │   └── llm_service.py
│   ├── qimen/            # 奇门相关服务
│   │   ├── __init__.py
│   │   ├── day_service.py
│   │   └── llm_service.py
│   ├── ziwei/            # 紫微相关服务
│   │   ├── __init__.py
│   │   ├── llm_service.py
│   │   ├── report_service.py
│   │   └── year_service.py
│   ├── summary/          # 总结服务
│   │   ├── __init__.py
│   │   └── summary_service.py
│   ├── year_score/       # 年运势评分服务
│   │   ├── __init__.py
│   │   └── score_service.py
│   ├── llm/              # LLM通用服务
│   │   ├── __init__.py
│   │   ├── llm_calls.py
│   │   ├── llm_response.py
│   │   └── prompt_manager.py
│   ├── session/          # 会话管理服务
│   │   ├── __init__.py
│   │   └── session_manager.py
│   └── validation/       # 验证服务
│       ├── __init__.py
│       ├── validation_rules.py
│       └── query_intent.py
│
├── schemas/              # 数据模型层 - Pydantic模型定义
│   ├── __init__.py
│   ├── leinuo.py         # 雷诺相关模型
│   ├── qimen.py          # 奇门相关模型
│   ├── ziwei.py          # 紫微相关模型
│   ├── summary.py        # 总结相关模型
│   ├── year_score.py     # 年运势评分模型
│   └── common.py         # 通用模型
│
├── database/             # 数据库层
│   ├── __init__.py
│   ├── db_manager.py     # 统一数据库管理
│   └── models.py         # 数据库模型
│
├── utils/                # 工具函数层
│   ├── __init__.py
│   ├── liushifunction.py # 紫微六十四卦函数
│   ├── tiangan_function.py # 天干地支函数
│   ├── xingyaoxingzhi.py # 星曜星质
│   ├── ziwei_ai_function.py # 紫微AI函数
│   ├── tokenizer.py      # 分词工具
│   ├── processing.py     # 数据处理工具
│   └── time_filters.py   # 时间过滤工具
│
├── clients/              # 外部客户端层
│   ├── __init__.py
│   ├── vllm_client.py    # VLLM客户端
│   ├── external_api_client.py # 外部API客户端
│   ├── astro_api_client.py # 占星API客户端
│   └── shared_client.py  # 共享客户端
│
├── prompts/              # 提示词配置
│   ├── __init__.py
│   ├── leinuo/
│   ├── qimen/
│   ├── ziwei/
│   ├── prompt_logic.py
│   ├── prompt_config_jili.py
│   └── prompt_config_jili2_year.py
│
├── security/             # 安全层
│   ├── __init__.py
│   └── verifier.py       # 签名验证
│
├── monitoring/           # 监控层
│   ├── __init__.py
│   └── monitor.py        # 监控工具
│
├── assets/               # 静态资源
│   ├── 卡牌vs牌号.csv
│   ├── 雷牌信息集合.csv
│   └── xyhs.xlsx
│
├── logs/                 # 日志目录
│
├── config.py             # 全局配置
├── main.py               # 应用入口
├── requirements.txt      # 依赖管理
├── .env                  # 环境变量
├── Dockerfile            # Docker配置
└── README.md             # 项目说明

```

## 项目映射关系

### 1. 10_10_leinuo_yunshi_day → routers/leinuo_day.py + services/leinuo/day_service.py
- 入口: api_main.py → routers/leinuo_day.py
- 业务逻辑保留在 services/leinuo/day_service.py

### 2. 11_5_yingshi_yunshi_year_score_ali → routers/year_score.py + services/year_score/score_service.py
- 入口: app/api.py → routers/year_score.py
- 业务逻辑: app/processing.py → services/year_score/score_service.py
- 工具: app/utils.py → utils/processing.py

### 3. 12_11_yingshi_yunshi_day_new_qimen → routers/qimen_day.py + services/qimen/day_service.py
- 入口: api_main_*.py → routers/qimen_day.py (多个端点)
- 业务逻辑: app/processing.py → services/qimen/day_service.py
- LLM调用: app/llm_calls.py → services/llm/llm_calls.py
- 提示词: app/prompt_manager.py → prompts/prompt_logic.py
- 数据库: helper_libs/db_manager_yingshi.py → database/db_manager.py
- 天干函数: helper_libs/tiangan_function.py → utils/tiangan_function.py

### 4. 12_12_llm_yingshi_leinuo_ali → routers/leinuo_llm.py + services/leinuo/llm_service.py
- 入口: main.py → routers/leinuo_llm.py
- 业务逻辑保留在 services/leinuo/llm_service.py
- 提示词: prompt_logic.py → prompts/prompt_logic.py
- 意图识别: queryIntent.py → services/validation/query_intent.py
- 会话管理: session_manager.py → services/session/session_manager.py
- 验证规则: validation_rules.py → services/validation/validation_rules.py

### 5. 12_12_llm_yingshi_qimen_ali → routers/qimen_llm.py + services/qimen/llm_service.py
- 入口: main.py → routers/qimen_llm.py
- 业务逻辑保留在 services/qimen/llm_service.py
- 数据库查询: db_query.py → services/qimen/llm_service.py (内部方法)
- LLM响应: llm_response.py → services/llm/llm_response.py
- 意图识别: query_intent.py → services/validation/query_intent.py
- 会话管理: session_manager.py → services/session/session_manager.py
- 用户信息提取: user_info_extractor.py → services/qimen/llm_service.py
- 验证规则: validation_rules.py → services/validation/validation_rules.py

### 6. 12_12_llm_yingshi_summary_ali → routers/summary.py + services/summary/summary_service.py
- 入口: app/main.py → routers/summary.py
- 业务逻辑保留在 services/summary/summary_service.py
- 意图识别: app/queryIntent.py → services/validation/query_intent.py
- 会话管理: app/session_manager.py → services/session/session_manager.py
- 时间过滤: app/time_filters.py → utils/time_filters.py
- 验证规则: app/validation_rules.py → services/validation/validation_rules.py

### 7. 12_12_llm_yingshi_ziwei_ali → routers/ziwei_llm.py + services/ziwei/llm_service.py
- 入口: api_main.py → routers/ziwei_llm.py
- 业务逻辑: services/chat_processor.py → services/ziwei/llm_service.py
- 紫微分析: services/ziwei_analyzer.py → services/ziwei/llm_service.py
- 数据库: database/db_manager.py → database/db_manager.py (保留)
- 客户端: clients/* → clients/* (保留)
- 安全: security/verifier.py → security/verifier.py (保留)
- 会话管理: services/session_manager.py → services/session/session_manager.py
- 意图识别: services/queryIntent.py → services/validation/query_intent.py
- 验证规则: services/validation_rules.py → services/validation/validation_rules.py
- 时间工具: services/chat_time_utils.py → utils/time_filters.py
- 监控: services/monitor.py → monitoring/monitor.py
- 模型: models.py → schemas/ziwei.py
- 提示词配置: prompt_cofig_jili.py → prompts/prompt_config_jili.py
- 提示词逻辑: prompt_logic.py → prompts/prompt_logic.py
- 工具函数: liushifunction.py, tiangan_function.py, utils.py, tokenizer.py, ziwei_ai_function.py → utils/

### 8. report_prediction_bendi_test_12_8 → routers/ziwei_report.py + services/ziwei/report_service.py
- 入口: api_main.py → routers/ziwei_report.py
- 业务逻辑保留在 services/ziwei/report_service.py
- 数据库: db_manager2.py → database/db_manager.py
- 工具函数: liushifunction.py, tiangan_function.py, utils.py, tokenizer.py, xingyaoxingzhi.py, ziwei_ai_function.py → utils/

### 9. ziwei_report_year_aliyun_11_12 → routers/ziwei_year.py + services/ziwei/year_service.py
- 入口: api_main.py → routers/ziwei_year.py
- 业务逻辑保留在 services/ziwei/year_service.py
- 占星API: astro_api_client.py → clients/astro_api_client.py
- 数据库: db_manager2.py → database/db_manager.py
- 提示词: prompt_config_jili2_year.py, prompts.py → prompts/
- 工具函数: liushifunction.py, tiangan_function.py, utils.py, tokenizer.py, xingyaoxingzhi.py, ziwei_ai_function.py → utils/

## 合并原则

1. **保留所有接口名**: 所有原有的API端点路径和名称保持不变
2. **保留业务逻辑**: 所有业务逻辑代码完整迁移，不做修改
3. **保留文件引用**: 所有import语句和文件引用更新为新的路径
4. **统一共享模块**: 相同功能的模块合并为一个，避免重复
5. **分层清晰**: Router只处理HTTP，Service处理业务，Schema定义模型
6. **配置统一**: 所有配置集中到config.py和.env
7. **依赖合并**: 所有requirements.txt合并去重

## 实施步骤

1. 创建新的目录结构
2. 合并共享模块（utils, database, clients等）
3. 迁移每个项目的路由和服务
4. 更新所有import路径
5. 合并配置文件和环境变量
6. 创建统一的main.py入口
7. 合并requirements.txt
8. 测试验证
