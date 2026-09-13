"""配置管理：从 .env 读取 API 配置"""
import os
from dotenv import load_dotenv

load_dotenv()

# Qwen API (DashScope OpenAI 兼容接口)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.7-flash")
# 快模型：结构化解析这类简单任务用（思考固定关，见 base.py）。当前与主模型同款——
# qwen3.7-flash 输入输出价低于旧 turbo，速度同档，指令遵循高一档
DASHSCOPE_FAST_MODEL = os.getenv("DASHSCOPE_FAST_MODEL", "qwen3.7-flash")
# 主 agent 思考模式（qwen3 系列思考型模型）：默认开——意图判断/工具选择错一次就丢数据，可靠性优先。
# sub-agent（parser 等）不受此开关控制，固定关闭思考：结构化任务只要快。
DASHSCOPE_ENABLE_THINKING = os.getenv("DASHSCOPE_ENABLE_THINKING", "true").lower() in ("1", "true", "yes")

# 向量召回（检索的内容匹配主路）：默认关——n-gram 规则路够用时不多养一条 API 依赖。
# 开启步骤：① .env 加 EMBEDDING_ENABLED=1（.env 改动不触发 --reload，需重启或 touch 本文件）
#          ② 跑 .venv/bin/python scripts/backfill_embeddings.py 回填存量记录
# 换 EMBEDDING_MODEL = 侧车向量全部作废（不同模型的向量不可互比），需重跑回填
EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "false").lower() in ("1", "true", "yes")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# 数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# 记忆衰减参数：幂律半衰期（天）——经过该天数强度减半
# 旧指数衰减在年尺度上会把强度压到 ~0（done 一年 1.5e-5），
# "去年今天"等长时程召回数学上不可达，见 docs/capacity-analysis.md 问题②
DECAY_HALF_LIVES = {
    "moment": 365,  # 高光时刻最难忘：一年才减半
    "mood": 180,
    "plan": 120,
    "done": 60,     # 日常琐事淡得最快
}

# 晨间总览的默认城市（与 interactive 的天气路由默认一致；想换城市时在 .env 加一行即可）
BRIEFING_CITY = os.getenv("BRIEFING_CITY", "北京")

# 高德 MCP（Streamable HTTP 远程托管，mcp.amap.com，官方推荐方式）—— 当前只接天气查询
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY", "")
MCP_AMAP_ENABLED = os.getenv("MCP_AMAP_ENABLED", "true").lower() in ("1", "true", "yes")
# key 拼在 URL 上：所有进日志的异常信息必须过 mcp_client 的 _scrub() 脱敏
AMAP_MCP_URL = f"https://mcp.amap.com/mcp?key={AMAP_MAPS_API_KEY}"
# 工具白名单：tools/list 动态发现的工具只注册名单内的，其余忽略（暂只接入天气）
MCP_AMAP_TOOL_WHITELIST = ["maps_weather"]
