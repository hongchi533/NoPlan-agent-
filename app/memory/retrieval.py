"""记忆检索：模式分派粗筛 + 素材直给（检索管线，无 LLM）

查询先按 有无时间 × 有无有效关键词 分派到四种模式：
  browse（时间全量）/ lookup（内容精准）/ intersect（时间∩内容）/ fallback（近30天）
时间归一化在交互层完成（search_memory 的 date_from/date_to 参数），
规则表（_extract_time_range）只做参数缺失时的兜底。
内容匹配主路默认 n-gram + IDF；EMBEDDING_ENABLED 开启后升级为向量余弦，
n-gram 退为旁路（app/memory/embeddings.py）。
数据量在千级以内，不使用向量数据库——侧车 JSON + 暴力扫描就够。

叙述权在交互 agent：本模块只做"选材"，把候选记录原文 + 习惯块 + 使用说明
直给最外层，由它单次叙述——中间不隔摘要 LLM（2026-09-04 看电视事故：
正解排候选第 1，两跳叙述后丢失，"上次看电视"被答成更早的日期）。
"""
import calendar
import math
import re
from datetime import date, datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union

from app.memory import embeddings
from app.memory.decay import should_surface_for_last_year, should_surface_for_capsule
from app.models.schemas import Event, MoodRecord
from app.store import db

logger = logging.getLogger(__name__)

# ── 模式分派参数 ──────────────────────────────────
BROWSE_MAX = 200   # browse 全量上限：一条记录 ≈20 tok，200 条 ≈4k tok（宁多勿漏的账）
LOOKUP_MAX = 30    # lookup 少而精："上次…"类只要近因若干条，多了稀释注意力
FLOOD_RATIO = 0.3  # 泛滥词线：命中超全库 30% 的 gram（'生活'式）不构成"有效关键词"
CAPSULE_COUNT = 3          # 时光胶囊一次最多唤起条数
CAPSULE_COOLDOWN_DAYS = 7  # 胶囊展示冷却：同一条 7 天内不重复推送


def _rank_by_score(ids: set, item_score: Dict[str, float],
                   id_to_item: Dict[str, Union[Event, MoodRecord]]) -> list:
    """gram 命中项排序：IDF 总分降序，同分近因优先（先按 date/strength 排，
    再稳定排序按分数——等分数内保持近者在前的次序）"""
    ranked = sorted((id_to_item[i] for i in ids if i in id_to_item),
                    key=lambda x: (x.date, x.strength), reverse=True)
    ranked.sort(key=lambda x: -item_score.get(x.id, 0.0))
    return ranked

# 素材使用说明：随素材拼接给叙述者（交互 agent）。今天锚点交互层已注入，这里
# 不重复；只放"怎么从素材取事实"的校准规则——时间锚点一条的来历：没有它时
# 模型分不清 plan 日期在未来，把计划当往事叙述（2026-09-01 实测"上一次跑步"
# 答成 9月3日的 plan）；末条是叙述篇幅约束（2026-09-13 用户定：回忆类 ≤150 字）
MATERIAL_GUIDE = """- 只根据下面的记录和习惯回答，没有相关信息就如实告知，不要编造
- 记录格式 [日期, 类型 时刻?, 标题, 备注?, 标签?]：done=已发生，plan=计划；
  日期晚于今天是还没发生的计划，今天的 plan 要看时刻是否已过。
  问"上一次/最近做过什么"只看今天之前的记录；未来的安排被问到才提，并说明是计划
- 习惯与记录是两种东西：习惯是规律（"一般/喜欢"），记录是某天真实发生/计划的事。
  答"一般/通常/喜欢"优先用习惯、可用记录佐证（一致或背离都如实说）；
  答"上次/某天"只依据记录。绝不把习惯的默认时间说成某天真实发生，也不把单次记录说成规律
- 日期、时刻、星期等事实以素材原文为准，不凭印象补全
- 回忆类回答 ≤150 字：挑印象深的几条说，记录多时按主题归纳，不逐条罗列清单；用户想细看自然会追问"""

WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


class MemoryRetrieval:
    """检索管线：模式分派粗筛 + 素材格式化（无 LLM，叙述权在交互 agent）"""

    async def search(self, query: str, date_from: str = "", date_to: str = "") -> str:
        """语义检索：粗筛候选 + 常驻习惯块 → 原文素材直给交互 agent

        Args:
            query: 用户原话查询（交互层照传，不提炼关键词——"上次/什么时候"
                这类措辞承载着叙述意图，n-gram 需要的清洗粗筛层自己做）
            date_from/date_to: 交互层换算好的绝对时间范围（YYYY-MM-DD，含端点）；
                不传时用规则兜底（_extract_time_range）

        Returns:
            素材文本：使用说明 + 习惯块 + 记录块
        """
        # 习惯类问题（"我一般几点吃午饭"）可能事件零命中而偏好正中，
        # 所以"没找到"的判定必须等偏好块也确认过
        candidates, mode = await self._coarse_retrieve(query, date_from, date_to)
        prefs_text = self._format_preferences()

        if not candidates and not prefs_text:
            return "暂时没有找到相关的记忆记录。"

        # 习惯块（规律）与记录块（事实）物理分块，不混排。
        # 素材头部带今天锚点：叙述者读素材时"今天"就在手边（指南里"日期晚于
        # 今天是计划"的判断直接可用），日志审计素材块也不用回翻 system prompt
        today = date.today()
        header = f"【今天】{today.isoformat()} {WEEKDAYS[today.weekday()]} {datetime.now().strftime('%H:%M')}"
        records_text = self._format_records(candidates) if candidates else "无"
        return (
            f"{header}\n\n"
            f"【使用说明】\n{MATERIAL_GUIDE}\n\n"
            f"【用户习惯（规律，不是发生过的事）】\n{prefs_text or '无'}\n\n"
            f"【记录（某天真实发生或计划中的事）】\n{records_text}"
        )

    async def _coarse_retrieve(self, query: str, date_from: str = "", date_to: str = "",
                               max_count: int = 50) -> Tuple[List[Union[Event, MoodRecord]], str]:
        """模式分派粗筛：browse / lookup / intersect / fallback 四路

        时间优先级：调用方参数（交互层按今天换算的绝对日期）> 规则兜底
        （_extract_time_range）。模式 = 有无时间 × 有无有效关键词（泛滥词不算，
        见 FLOOD_RATIO）；判错的落点都宁多勿漏——intersect 交集空退化（关键词
        全局有命中退 lookup、纯措辞噪声退 browse），browse 全量上限 BROWSE_MAX。
        向量开关（EMBEDDING_ENABLED，默认关）开启时 lookup/intersect 的内容
        匹配主路是 embedding 余弦；n-gram 退为旁路：补没有向量的记录 +
        向量路任何故障（disabled/stale/empty/failed）时整体接管。
        """
        events = db.load_events()
        moods = db.load_moods()

        id_to_item = {}
        for e in events:
            id_to_item[e.id] = e
        for m in moods:
            id_to_item[m.id] = m
        n_total = len(id_to_item)

        # ① 时间范围：参数优先，规则兜底（直调/参数缺失路径），双轨校验记日志
        rng_from, rng_to, time_src = self._resolve_range(query, date_from, date_to)

        # ② 关键词：gram 命中表 + IDF——一次计算两处用（排序权重 & 泛滥词判定）。
        #    稀有 gram（咖啡）权重大、泛滥 gram（生活）权重小，BM25 思想的零依赖版
        grams = self._extract_keywords(query)
        gram_hits: Dict[str, set] = {}
        for g in grams:
            hits = set()
            for e in events:
                if g in e.source_text or g in (e.note or "") or g in e.title:
                    hits.add(e.id)
            for m in moods:
                if g in (m.content or ""):
                    hits.add(m.id)
            gram_hits[g] = hits
        item_score: Dict[str, float] = {}
        for g, ids in gram_hits.items():
            weight = math.log(n_total / (1 + len(ids))) if n_total else 0.0
            for i in ids:
                item_score[i] = item_score.get(i, 0.0) + weight
        effective = [g for g in grams if len(gram_hits[g]) <= FLOOD_RATIO * n_total]
        kw_ids = set(item_score.keys())

        # ③ 标签桶：query 推断标签 → 带该标签的事件（语义补充，不参与 IDF 排名）
        tags = self._infer_tags(query)
        tag_ids = set()
        for tag in tags:
            for e in events:
                if tag in e.tags:
                    tag_ids.add(e.id)

        has_time = bool(rng_from and rng_to)
        has_kw = bool(effective)

        if has_time and not has_kw:
            # browse：时间界定要全量（"上个月我都干了什么"），按时间线排给 LLM 叙述
            mode = "browse"
            picked = sorted(
                (it for it in id_to_item.values() if rng_from <= it.date <= rng_to),
                key=lambda x: (x.date, getattr(x, "start_time", None) or ""),
            )[:BROWSE_MAX]
            detail = f"范围内{len(picked)}条"
            if len(picked) == BROWSE_MAX:
                logger.warning(f"[retrieval] browse 触到上限 {BROWSE_MAX} 条——数据密度已到分层记忆的启用线")

        elif not has_time and has_kw:
            # lookup：内容界定要精准（"上次喝咖啡是什么时候"），近因优先
            mode = "lookup"
            vec_scores, vec_status = await embeddings.query_top(
                query, set(id_to_item), LOOKUP_MAX)
            picked = self._lookup_pick(vec_scores, vec_status, item_score,
                                       tag_ids, id_to_item, LOOKUP_MAX)
            detail = f"向量[{vec_status}]"

        elif has_time and has_kw:
            # intersect：时间 ∩ 内容（"上个月和小王吃了什么"）
            mode = "intersect"
            window_ids = {i for i, it in id_to_item.items()
                          if rng_from <= it.date <= rng_to}
            vec_scores, vec_status = await embeddings.query_top(
                query, window_ids, max_count)
            if vec_status == "ok":
                # 向量主路：窗口内按余弦排；gram/tag 命中里没向量的由旁路补位
                vec_map = dict(vec_scores)
                ordered = [id_to_item[i] for i, _ in vec_scores]
                rest = (window_ids & (kw_ids | tag_ids)) - set(vec_map)
            else:
                ordered = []
                rest = window_ids & (kw_ids | tag_ids)
            picked = (ordered + _rank_by_score(rest, item_score, id_to_item))[:max_count]
            if not picked:
                # 交集空：关键词全局有命中 → 时间可能挡住了答案，退 lookup 全局找；
                # 全局也无命中 → 纯措辞噪声（"3月有哪些值得记录的事情"式），退 browse 全量筛
                if kw_ids:
                    mode = "lookup*"
                    picked = self._lookup_pick([], "failed", item_score,
                                               tag_ids, id_to_item, LOOKUP_MAX)
                else:
                    mode = "browse*"
                    picked = sorted((id_to_item[i] for i in window_ids),
                                    key=lambda x: (x.date, getattr(x, "start_time", None) or ""))[:BROWSE_MAX]
                logger.info(f"[retrieval] intersect 交集空，退化 {mode}")
            detail = f"窗口{len(window_ids)}条 向量[{vec_status}]"

        else:
            # fallback：既无时间也无有效关键词，最近 30 天兜底（旧行为）
            mode = "fallback"
            start = (date.today() - timedelta(days=30)).isoformat()
            picked = sorted((it for it in id_to_item.values() if it.date >= start),
                            key=lambda x: x.strength, reverse=True)[:max_count]
            logger.info("[retrieval] 无时间无有效关键词，退回最近30天窗口兜底")
            detail = "近30天"

        logger.info(
            f"[retrieval] 粗筛 mode={mode} 时间={time_src}[{rng_from}~{rng_to}] "
            f"grams={len(grams)}(有效{len(effective)}) 标签{tags} {detail} 选中{len(picked)}"
        )
        return picked, mode

    @staticmethod
    def _lookup_pick(vec_scores, vec_status, item_score, tag_ids, id_to_item, cap):
        """lookup 选席：向量主路（余弦降序）→ n-gram 旁路补位（IDF 降序）→ 标签补充

        向量路任何不畅都整体让位 n-gram——这就是"开关一开 n-gram 自然变成
        降级旁路"的落点；补位而非替换，是因为没向量的记录（写入时 embedding
        失败的）仍然要被找得到。
        """
        picked = []
        chosen = set()
        if vec_status == "ok":
            for i, _ in vec_scores:
                if i in id_to_item and i not in chosen:
                    picked.append(id_to_item[i])
                    chosen.add(i)
        for it in _rank_by_score(set(item_score) - chosen, item_score, id_to_item):
            if len(picked) >= cap:
                break
            picked.append(it)
            chosen.add(it.id)
        for it in sorted((id_to_item[i] for i in tag_ids if i in id_to_item and i not in chosen),
                         key=lambda x: (x.date, x.strength), reverse=True):
            if len(picked) >= cap:
                break
            picked.append(it)
            chosen.add(it.id)
        return picked[:cap]

    def _resolve_range(self, query: str, date_from: str,
                       date_to: str) -> Tuple[Optional[str], Optional[str], str]:
        """时间范围合一：参数（交互层换算）优先 > 规则兜底（_extract_time_range）

        返回 (from, to, 来源标签 param/rule/none)。只传一端按单日；from>to 交换；
        双轨校验——参数与规则都算得出且不一致时 warning（观察 LLM 换算质量用，
        不进逻辑，以参数为准）。
        """
        pf = self._normalize_date_param(date_from, "date_from")
        pt = self._normalize_date_param(date_to, "date_to")
        if pf and not pt:
            pt = pf
        if pt and not pf:
            pf = pt
        rf, rt = self._extract_time_range(query)
        if pf and pt:
            if pf > pt:
                logger.warning(f"[retrieval] 时间参数反了（{pf} > {pt}），已交换")
                pf, pt = pt, pf
            if rf and rt and (rf, rt) != (pf, pt):
                logger.warning(f"[retrieval] 双轨校验不一致：参数 {pf}~{pt} vs 规则 {rf}~{rt}（以参数为准）")
            return pf, pt, "param"
        if rf and rt:
            return rf, rt, "rule"
        return None, None, "none"

    @staticmethod
    def _normalize_date_param(s: str, label: str) -> Optional[str]:
        s = (s or "").strip()
        if not s:
            return None
        try:
            return date.fromisoformat(s[:10]).isoformat()
        except ValueError:
            logger.warning(f"[retrieval] {label} 参数格式不对：{s}（要 YYYY-MM-DD），已忽略走规则兜底")
            return None

    def _extract_time_range(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """从查询中提取时间范围（规则匹配：绝对日期优先，其次相对时间词）"""
        today = date.today()

        # 绝对日期（比相对词更具体，先试）：
        #   2026年8月15日 → 当天 / 2026年8月 → 整月 / 8月15日 与 8月 → 无年份时按
        #   "最近的过去那个月"归位（9月问12月=去年12月；1月问12月也是），去年前缀强制减一年。
        #   要查未来月份的计划，带年份说"今年12月"，或走具体某天的 get_overview
        m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]?", query)
        if m:
            y, mo, d = map(int, m.groups())
            try:
                day = date(y, mo, d)
                return (day.isoformat(), day.isoformat())
            except ValueError:
                pass
        m = re.search(r"(\d{4})年(\d{1,2})月", query)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12:
                last = calendar.monthrange(y, mo)[1]
                return (date(y, mo, 1).isoformat(), date(y, mo, last).isoformat())
        m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", query)
        if m:
            mo, d = int(m.group(1)), int(m.group(2))
            y = (today.year - 1 if "去年" in query else
                 today.year if (mo, d) <= (today.month, today.day) else today.year - 1)
            try:
                day = date(y, mo, d)
                return (day.isoformat(), day.isoformat())
            except ValueError:
                pass
        m = re.search(r"(\d{1,2})月", query)
        if m:
            mo = int(m.group(1))
            if 1 <= mo <= 12:
                y = today.year - 1 if "去年" in query or mo > today.month else today.year
                last = calendar.monthrange(y, mo)[1]
                return (date(y, mo, 1).isoformat(), date(y, mo, last).isoformat())

        if "去年" in query:
            return (today.replace(year=today.year - 1).isoformat(), today.isoformat())
        # 上上系列必须在单上系列之前判断："上上个月"包含子串"上个月"，
        # 顺序反了会被单上分支抢走，静默错窗一个月（2026-09-04 修）
        if "上上个月" in query:
            mo, y = today.month - 2, today.year
            if mo < 1:
                mo += 12
                y -= 1
            last = calendar.monthrange(y, mo)[1]
            return (date(y, mo, 1).isoformat(), date(y, mo, last).isoformat())
        if "上个月" in query:
            prev = today.replace(day=1) - timedelta(days=1)
            start = prev.replace(day=1).isoformat()
            end = prev.isoformat()
            return (start, end)
        if "这个月" in query or "本月" in query:
            start = today.replace(day=1).isoformat()
            return (start, today.isoformat())
        if "最近" in query:
            start = (today - timedelta(days=7)).isoformat()
            return (start, today.isoformat())
        if "这周" in query or "本周" in query:
            start = (today - timedelta(days=today.weekday())).isoformat()
            return (start, today.isoformat())
        if "上上周" in query:   # 同上上个月：必须在"上周"之前（子串包含）
            start = today - timedelta(days=today.weekday() + 14)
            end = start + timedelta(days=6)
            return (start.isoformat(), end.isoformat())
        if "上周" in query:
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
            return (start.isoformat(), end.isoformat())
        if "下周" in query:
            start = today + timedelta(days=7 - today.weekday())
            end = start + timedelta(days=6)
            return (start.isoformat(), end.isoformat())
        if "大前天" in query:   # 先于"前天"（子串包含）
            d = (today - timedelta(days=3)).isoformat()
            return (d, d)
        if "前天" in query:
            d = (today - timedelta(days=2)).isoformat()
            return (d, d)
        if "今天" in query:
            return (today.isoformat(), today.isoformat())
        if "昨天" in query:
            y = (today - timedelta(days=1)).isoformat()
            return (y, y)

        # 无明确时间词：不启用窗口桶，走"全时段 + 纯关键词"。
        # （过去向查询如"我以前…"套默认近期窗口，会把候选席灌满近期记录，
        #   挤掉关键词精准命中的老记忆——见 docs/capacity-analysis.md 问题①）
        # 三桶皆空的兜底在 _coarse_retrieve 里处理。
        return (None, None)

    def _extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词（n-gram 切分，n=2/3，不产生单字）

        整串子串匹配要求连续命中（"练吉他"匹配不上"练了半小时吉他"），
        n-gram 放宽为任意 2~3 字连续片段命中，解决跳字问题。
        清洗后不足 2 字则放弃关键词路（单字匹配噪声太大）。
        清洗质量直接决定模式分派：残留伪关键词会把 browse 误判成 intersect
        （"上个月我都干了什么"必须洗到只剩噪声），所以长词优先删除——
        set 迭代无序，"做了什么"若晚于"什么"删除会残留"做了"伪关键词。
        """
        time_words = {"去年", "上上个月", "上个月", "这个月", "本月", "上上周", "最近",
                       "这周", "本周", "上周", "下周", "大前天", "前天", "今天", "昨天",
                       "以前", "曾经", "做过什么", "做了什么", "干了什么", "我都",
                       "上次", "什么", "时候", "怎么样", "有没有",
                       "还", "吗", "了", "的", "和", "跟"}
        cleaned = query
        for tw in sorted(time_words, key=len, reverse=True):
            cleaned = cleaned.replace(tw, "")
        # 绝对日期表达整段剔除：时间归窗口桶管，留在关键词路只会被切成
        # '20''02' 这类数字碎片（纯噪声，还会误匹配"跑了20分钟"式文本）
        cleaned = re.sub(r"\d{4}年\d{1,2}月(\d{1,2}[日号]?)?|\d{1,2}月\d{1,2}[日号]|\d{1,2}月", "", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) < 2:
            logger.info(f"[retrieval] 关键词提取: 清洗后='{cleaned}'（不足2字，放弃关键词路）")
            return []
        grams = []
        for n in (2, 3):
            for i in range(len(cleaned) - n + 1):
                g = cleaned[i:i + n]
                if g not in grams:
                    grams.append(g)
        logger.info(f"[retrieval] 关键词提取: 清洗后='{cleaned}' → grams={grams}")
        return grams

    def _infer_tags(self, query: str) -> List[str]:
        """从查询中推断可能相关的标签"""
        tag_keywords = {
            "工作": ["工作", "上班", "加班", "开会", "周报", "项目"],
            "运动": ["运动", "跑步", "健身", "游泳", "打球", "锻炼"],
            "生活": ["生活", "家", "收拾", "整理", "打扫", "做饭"],
            "社交": ["朋友", "聚会", "吃饭", "约", "见面"],
            "饮食": ["吃", "喝", "咖啡", "外卖", "做饭", "餐厅"],
            "学习": ["学", "看", "读书", "课程", "练习"],
        }
        tags = []
        for tag, kws in tag_keywords.items():
            if any(kw in query for kw in kws):
                tags.append(tag)
        return tags

    def _format_records(self, items: List[Union[Event, MoodRecord]]) -> str:
        """格式化候选记录为文本，直给交互 agent 叙述

        日期带星期：叙述层爱补"（上周五）"式修饰，星期不能让它猜——不注入
        它就编（2026-09-04 实测 09-02 周二被说成"上周五"；get_overview 同款教训）
        """
        lines = []
        for i, item in enumerate(items, 1):
            wd = WEEKDAYS[date.fromisoformat(item.date).weekday()]
            if isinstance(item, Event):
                tags_str = ", ".join(item.tags) if item.tags else ""
                note_str = f", {item.note}" if item.note else ""
                time_str = f" {item.start_time}" if item.start_time else ""
                lines.append(
                    f"{i}. [{item.date} {wd}, {item.type}{time_str}] {item.title}{note_str}  [{tags_str}]"
                )
            elif isinstance(item, MoodRecord):
                from app.models.schemas import MOOD_EMOJI
                emoji = MOOD_EMOJI.get(item.mood, "😐")
                content_str = f", {item.content}" if item.content else ""
                lines.append(f"{i}. [{item.date} {wd}, mood] {emoji}{content_str}")
        return "\n".join(lines)

    def _format_preferences(self) -> str:
        """偏好 → 习惯块文本（规律措辞，供检索 LLM 与记录对照取用）

        与 parser 的偏好渲染意图不同：这里要"一般在 X / 喜欢 Y"的规律句式，
        杜绝裸时间戳——记录行的时间带全日期（发生格式），习惯行的时间裹在
        "一般在"里（规律格式），两种措辞互不伪装。偏好个位数常驻全量注入
        （零额外轮次），涨到几十条再套事件侧同款粗筛。
        """
        try:
            prefs = db.load_preferences()
        except Exception as e:
            logger.warning(f"[retrieval] 读取偏好失败: {e}")
            return ""

        lines = []
        for p in prefs:
            parts = []
            if p.rules.get("default_time"):
                parts.append(f"一般在 {p.rules['default_time']}")
            if p.rules.get("fact"):
                parts.append(p.rules["fact"])
            if not parts:
                continue  # 只剩 default_tag 的旧偏好（已废弃字段）无规律可述
            source_label = "你自己说过" if p.source == "manual" else "从记录里发现的"
            lines.append(f"- {p.pattern}：{'，'.join(parts)}（{source_label}）")
        return "\n".join(lines)


# ─── 召回：被动触发 ───────────────────────────────────

def get_last_year_today() -> List[Union[Event, MoodRecord]]:
    """获取"去年今天"的记忆（门槛见 decay.should_surface_for_last_year）"""
    target = (date.today() - timedelta(days=365)).isoformat()
    events = db.get_events_by_date(target)
    moods = db.get_moods_by_date(target)
    # 只展示强度够的（门槛集中在 decay.py，避免内联硬编码与函数脱节）
    events = [e for e in events if should_surface_for_last_year(e.strength)]
    moods = [m for m in moods if should_surface_for_last_year(m.strength)]
    return events + moods


def get_time_capsule_candidates() -> List[Event]:
    """获取适合被时光胶囊唤起的记忆（唤起区间见 decay.should_surface_for_capsule）

    选材：唤起窗内按强度降序取前 CAPSULE_COUNT 条——刚进窗的（强度高）先推，
    推过的自然沉后。展示冷却 CAPSULE_COOLDOWN_DAYS 天：同一条 7 天内不重复推；
    当天内重复调用不受影响（今天已展示的仍可选中，刷新页面卡片不闪没）。
    选中即把"今天"记入 surface_log 侧车（展示侧记账，不进 events.json）。
    """
    events = db.load_events()
    band = [e for e in events if should_surface_for_capsule(e.strength)]
    band.sort(key=lambda e: e.strength, reverse=True)

    log = db.load_surface_log()
    today = date.today()
    today_str = today.isoformat()

    def _eligible(e: Event) -> bool:
        last = log.get(e.id)
        if last is None or last == today_str:
            return True  # 没展示过 / 今天已展示（当天稳定在卡上）
        try:
            return (today - date.fromisoformat(last)).days >= CAPSULE_COOLDOWN_DAYS
        except ValueError:
            return True  # 脏日期不当拦路虎，宁可多推一次

    picked = [e for e in band if _eligible(e)][:CAPSULE_COUNT]
    for e in picked:
        log[e.id] = today_str
    if picked:
        db.save_surface_log(log)
    return picked
