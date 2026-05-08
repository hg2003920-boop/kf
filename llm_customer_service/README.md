# LLM 智能客服系统 (atguigu_ai)

这是一个基于大语言模型（LLM）驱动的智能对话架构精简实现，主要用于教学和演示目的。系统集成了一套完整的对话系统框架，支持自然语言理解、对话管理策略控制和知识检索，并包含了一个电商客服演示项目（`ecs_demo`）。

## 🌟 核心特性

- **大模型驱动**：深度集成 LangChain 和 LangGraph，支持对接主流大语言模型（如 OpenAI、通义千问/DashScope等）。
- **知识增强检索 (GraphRAG)**：支持基于 Neo4j 图数据库的知识图谱检索（知识抽取和问答）。
- **模块化设计**：包含 agent、API 服务接口、对话流程控制（core）、策略（policies）、NLU 理解、NLG 生成和检索模块。
- **配套示例项目**：内置电商客服 demo (`ecs_demo`)，提供端点配置参考和实践示例。

## 📂 目录结构

```text
llm_customer_service/
├── atguigu_ai/                  # 核心对话系统框架源码包
│   ├── agent/                   # 智能体核心逻辑
│   ├── api/                     # FastAPI Web服务接口
│   ├── channels/                # 多渠道接入层
│   ├── cli/                     # 命令行工具实现
│   ├── core/                    # 核心数据模型和流程引擎
│   ├── dialogue_understanding/  # NLU 对话理解模块
│   ├── nlg/                     # 自然语言生成模块
│   ├── policies/                # 路由和对话策略控制
│   └── retrieval/               # 检索模块（支持向量检索、图数据库等）
├── ecs_demo/                    # 电商智能客服演示项目
│   ├── config.yml               # 核心配置文件（定义流程、策略等）
│   ├── endpoints.yml            # 基础设施端点配置（LLM、Neo4j、MySQL 连接信息）
│   ├── gen_data.py              # 测试数据生成与初始化脚本
│   └── .env                     # 环境变量配置文件（用于存放 API Key 等敏感信息）
├── requirements-atguigu.txt     # 项目核心依赖列表
├── setup.py                     # 项目安装配置脚本
└── test_env.py                  # 环境测试脚本
```

## 🛠️ 技术栈

- **Web 框架**: FastAPI, Uvicorn
- **大模型生态**: LangChain, LangGraph, OpenAI SDK, DashScope (阿里云)
- **图数据库 & 向量库**: Neo4j, neo4j-graphrag
- **关系型数据库**: MySQL, SQLAlchemy, PyMySQL
- **自然语言处理**: sentence-transformers (本地嵌入), jieba (中文分词)

## 🚀 快速开始

### 1. 环境准备

建议使用 Python `3.10` 及以上版本。推荐使用虚拟环境：

```bash
conda create -n llm_cs python=3.10
conda activate llm_cs
```

### 2. 安装项目

在项目根目录下执行以下命令，使用开发模式（editable）进行安装。修改源码可立即生效：

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

此命令会自动读取 `requirements-atguigu.txt` 安装相关依赖，并将 `atguigu_ai` 包及 `atguigu` 命令注册到你的 Python 环境中。

### 3. 配置演示项目 (ecs_demo)

进入示例目录，配置相关环境变量：

```bash
cd ecs_demo
```

确保同级目录下有一个 `.env` 文件，并在其中填写必要的凭证信息（参考 `endpoints.yml` 的占位符）：
```env
DASHSCOPE_API_KEY=你的通义千问API_KEY
NEO4J_PASSWORD=你的Neo4j密码
MYSQL_PASSWORD=你的MySQL密码
```

### 4. 常用命令参考

安装完成后，可以在命令行使用系统注册的全局命令进行操作（具体支持基于 `atguigu_ai.cli` 实现）：

- `atguigu init`: 初始化项目结构
- `atguigu train`: 训练/准备模型数据
- `atguigu run`: 启动应用服务
- `atguigu inspect`: 启动交互式测试终端

## 💡 常见问题

**Q: 为什么使用 `pip install -e .` 而不是直接运行 Python 脚本？**
A: `-e` 为 editable（可编辑）模式。安装后修改源码不仅能立即生效（适合开发阶段），同时系统会自动处理复杂的模块导入路径，并且全局注册 `atguigu` 命令，避免了手动设置 `PYTHONPATH` 的麻烦。
