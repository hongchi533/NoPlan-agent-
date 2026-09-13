# 记忆助手（myNoPlan）

一个基于多 Agent 架构的个人记忆助手：像聊天一样用一句话记录生活（做了什么 / 要做什么 / 心情 / 习惯），
支持自然语言回忆检索、日程管理、记忆衰减（幂律遗忘曲线）和主动推送（时光胶囊 / 去年今天 / 晨间总览 / 周报）。

它不是待办清单——待办应用总在提醒你"还没做完什么"，这个产品反着来：
让每条记忆按"印象深浅"而不是时间排序，过段时间回头一看，"原来我经历了这么多"。

## 架构概览

```
浏览器 UI（原生 HTML/JS/CSS：流式聊天 + 月/周/日历视图 + 提醒页）
        │ HTTP + SSE
        ▼
FastAPI + uvicorn（路由 / 全局异常兜底 / lifespan 生命周期管理）
        ▼
InteractiveAgent（交互 Agent：Agent Loop ≤8 轮，唯一对用户说话的出口）
  ├── 基础工具      确定性代码：查日程 / 完成计划 / 注册任务 / 今日总览（无 LLM）
  ├── ParserAgent   解析子 Agent：一句话 → 结构化事件/心情/偏好（快模型分槽，思考关）
  ├── 检索层        四模式分派（时间窗/关键词/交集/兜底）+ 素材直给（纯代码，无 LLM；
  │                 向量召回可选旁挂，DashScope embedding，坏了自动降级关键词路）
  └── MCP 动态工具  高德天气（Streamable HTTP 连 mcp.amap.com，白名单只挂 maps_weather）
        ▼
JSON 文件存储（data/*.json：原子写 + 文件锁 + 损坏隔离，无数据库）

Scheduler（统一调度器：每 60s 醒来对墙上时钟，三源合并——内置任务 + 用户任务 + 日程派生提醒）
   └─ 到点 fire → BackgroundAgent（夜间记忆整理 / 晨览 / 周报 / 自然语言委托）→ 信箱 outbox
        ▲
前端每 30s 轮询领取（确认式投递：领取-展示-签收 + 超时重领 + 保鲜期 = 恰好一次展示）
```

几条核心设计原则：

- **事实-计算-叙述三层分离**：events/moods 是唯一的事实账本；印象分、向量、总览、回忆卡片
  全是可重算的派生数据（放侧车文件，坏了删掉重算即可）；所有对用户说的话收敛到交互 Agent 一个出口。
- **选材归代码，说话归模型**：检索层无 LLM，把候选记录原文 + 使用须知整包交给交互 Agent 单次叙述，
  中间不隔任何摘要模型（消除多轮 LLM 的信息损耗与延迟）。
- **缩小爆炸半径**：任何单点故障只降级对应功能——嵌入服务挂了走关键词路、高德连不上只是天气缺位、
  LLM 断网有降级文案、未知异常统一 200 + ok:false，前端永远见不到 500。

## 环境要求

- Python 3.10+（mcp SDK 要求；用 [uv](https://docs.astral.sh/uv/) 管理最省事）
- DashScope（阿里云百炼）API Key —— 必填
- 高德开放平台 API Key —— 可选，仅天气查询用，类型必须是 **Web服务**

## 快速开始

### 1. 配置 API Key

```bash
cp .env.example .env    # 然后编辑 .env，填入 DASHSCOPE_API_KEY（必填）等
```

### 2. 安装依赖（uv + venv，Python 3.12）

```bash
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple uv   # 镜像加速
cd myNoPlan
uv python install 3.12                 # 若本机无 3.10+ 解释器
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 3. 启动服务器

```bash
cd myNoPlan
mkdir -p log
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload >> log/server.log 2>&1 &
```

- `--reload`：代码热重载（**只监听 .py 文件**——改 .env 后需 `touch app/config.py` 触发重载）
- `>> log/server.log 2>&1`：日志重定向到文件（所有 LLM 调用的耗时/token/降级记录都在这里）

启动后确认 / 看日志 / 停止：

```bash
curl -s localhost:8000/api/init | head -c 100    # 确认：有 JSON 返回即正常
tail -f log/server.log                           # 实时看日志（Ctrl+C 只退出查看，不影响服务）
pkill -f "uvicorn app.main"                      # 停止（优雅退出，日志里能看到 shutdown 逆序清理）
```

前台运行（调试用，日志直接看终端）：

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 访问

浏览器打开 <http://localhost:8000>，直接开始聊天记录。首次启动 data/ 为空是正常的，
所有数据文件会在使用中自动创建。

## 配置说明

配置分两层：`.env` 存值（含密钥，不进 git），`app/config.py` 负责读取和兜底默认。
换模型只改 `.env`，代码零改动；改后需重启或 touch 触发重载（模型名在进程启动时冻结）。

### LLM 配置

LLM 走 DashScope 的 OpenAI 兼容接口（通义千问 qwen 系列），按 Agent 分槽配置：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DASHSCOPE_API_KEY` | （必填） | 百炼 API Key |
| `DASHSCOPE_BASE_URL` | DashScope 兼容接口 | OpenAI 兼容 base url，兼容任何 OpenAI 协议的服务 |
| `DASHSCOPE_MODEL` | qwen3.7-flash | **主 agent（交互）**：意图判断 / 工具选择 / 最终叙述，思考模式默认开 |
| `DASHSCOPE_FAST_MODEL` | qwen3.7-flash | **解析 sub-agent**：结构化抽取，要快和准，思考固定关 |
| `DASHSCOPE_ENABLE_THINKING` | true | 主 agent 思考模式开关 |

主/子 Agent 分槽的意义：交互岗要模型"说话体贴、守规矩"，解析岗要"快而准"，两者对模型的要求相反，
所以各配各的换型互不影响。两槽当前同款是实测后的选择（该型号在说话岗的对比中胜出、解析岗时延也达标）。

### 向量与 MCP 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `EMBEDDING_ENABLED` | false | 向量召回开关；开启后需跑 `scripts/backfill_embeddings.py` 回填 |
| `EMBEDDING_MODEL` | text-embedding-v3 | 嵌入模型；换模型 = 旧向量全部作废，需重跑回填 |
| `AMAP_MAPS_API_KEY` | （空） | 高德 Web服务 Key；为空则 MCP 天气工具不挂载，其余功能不受影响 |
| `MCP_AMAP_ENABLED` | true | 高德 MCP 开关 |
| `BRIEFING_CITY` | 北京 | 晨间总览 / 天气查询的默认城市 |

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | 聊天入口（SSE 流式：状态/增量/收尾事件直通前端） |
| POST | `/api/chat` | 非流式入口（调试/兼容用） |
| GET | `/api/init` | 首页初始化（今日/未来事件、心情、回忆卡片、今日总览） |
| GET | `/api/overview` | 今日总览（页面开着时轮询） |
| GET | `/api/calendar?view=month\|week\|day&d=日期` | 日历视图数据 |
| GET/POST/DELETE | `/api/tasks` | 用户定时任务（提醒页表单与对话工具共用入口） |
| GET | `/api/reminders/due` | 前端 30s 信箱轮询（确认式投递） |
| POST | `/api/reminders/ack` | 前端展示成功后签收 |
| DELETE | `/api/events/{id}` · POST `/api/events/{id}/complete` | 手动删除 / 打勾完成（不过 agent loop） |
| DELETE | `/api/moods/{id}` | 手动删除心情 |
| POST | `/api/scheduler/run/{job_id}` | 手动触发任意任务（开发/演示） |

## 项目结构

```
myNoPlan/
├── app/
│   ├── main.py             # FastAPI 入口：路由 + lifespan 生命周期 + 全局异常兜底
│   ├── config.py           # 配置读取（.env → 环境变量 → 兜底默认）
│   ├── scheduler.py        # 统一调度器：三源合并 / 二分记账 / 信箱投递
│   ├── agent/
│   │   ├── interactive.py  # 交互 agent：Agent Loop + 状态机 + 降级 + SSE 事件
│   │   ├── background.py   # 后台 agent：夜间整理 / 晨览 / 周报 / 委托执行
│   │   ├── base.py         # 基类：LLM 调用 + 重试退避 + 模型分槽（model_override）
│   │   ├── parser.py       # 解析 agent：一句话 → 结构化记录（含四条代码级铁律）
│   │   ├── mcp_client.py   # MCP 客户端：连接/白名单/重连/超时/脱敏
│   │   └── tools.py        # 工具集（Function Calling 定义 + 动态注册）
│   ├── memory/
│   │   ├── retrieval.py    # 检索：四模式分派 + 素材直给（无 LLM）
│   │   ├── decay.py        # 衰减：幂律半衰期（四类记忆各不同）
│   │   └── embeddings.py   # 向量侧车：批量入库 / 余弦扫描 / 换模型自动作废
│   ├── holidays.py         # 节假日数据（晨览素材）
│   ├── models/schemas.py   # Pydantic 数据模型
│   └── store/db.py         # JSON 存储：原子写 + 文件锁 + 损坏隔离
├── ui/                     # 前端（index.html / app.js / style.css / journal.css）
├── scripts/                # 回填与评测脚本（向量回填 / 模型对比基准）
├── data/                   # 运行时数据（不进 git，自动生成）
├── log/                    # 运行日志（不进 git）
└── requirements.txt
```

## 常见问题

- **改了 .env 不生效** → `--reload` 只监听 .py：`touch app/config.py` 或重启
- **启动报 `no such file or directory: log/server.log` / 前端连不上** → 不在项目目录里，
  `cd myNoPlan` 后再执行；用 `lsof -i :8000` 确认端口有没有人监听——没人监听就是没启动成功
- **LLM 请求失败/断网** → 已内置多层兜底（客户端超时 30s → 重试退避 → agent 降级 →
  FastAPI 全局异常 → 前端超时），服务不会 500/崩溃，回复会提示"这条还没记住"
- **必须用 .venv 启动** → 系统 python3 若低于 3.10 没有 mcp SDK；
  用旧解释器启动只会降级（天气不可用），服务本身不报错
- **想清空数据重来** → 停服后删掉 data/ 下的 json 文件即可，下次启动自动从空数据开始
