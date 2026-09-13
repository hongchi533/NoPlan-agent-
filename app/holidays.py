"""法定节假日数据面（chinesecalendar 离线数据，无网络、无 key）

一面实现、两面使用：
- background.run_overview 直接调 upcoming_festivals() 拿晨报素材行（编排式，素材必到）
- tools.holiday_info 包装成对话工具（"中秋放几天假""下周六要调休吗"）

数据只覆盖到国务院已公布安排的年份（当前到 2026），越界抛 NotImplementedError：
按"素材缺失不致命"原则处理——晨报静默少一行，对话里如实告知。
"""
from datetime import date, timedelta
from typing import List, Optional, Tuple

from chinese_calendar import get_holiday_detail

# 包里的节日名是英文串，中文归属这张表（法定 7 项；2015 抗战 70 周年属历史一次性，不译）
_NAME_CN = {
    "New Year's Day": "元旦",
    "Spring Festival": "春节",
    "Tomb-sweeping Day": "清明",
    "Labour Day": "劳动节",
    "Dragon Boat Festival": "端午",
    "Mid-autumn Festival": "中秋",
    "National Day": "国庆",
}

# 口语关键词 → 英文名（双向包含匹配："端午节"含"端午"即可命中）
_ALIAS = {
    "元旦": "New Year's Day",
    "春节": "Spring Festival", "过年": "Spring Festival",
    "清明": "Tomb-sweeping Day",
    "劳动节": "Labour Day", "五一": "Labour Day",
    "端午": "Dragon Boat Festival",
    "中秋": "Mid-autumn Festival",
    "国庆": "National Day", "十一": "National Day",
}

_WEEKDAY = "一二三四五六日"


def _cn(name) -> str:
    return _NAME_CN.get(str(name), str(name))


def _span_days(first: date, name: str) -> int:
    """从假期首日向后数连续放假日（同名假期块长度）"""
    span = 1
    while True:
        try:
            hol, n = get_holiday_detail(first + timedelta(days=span))
        except NotImplementedError:
            break
        if not hol or n != name:
            break
        span += 1
    return span


def _block_start(d: date, name: str) -> date:
    """站在假期块中间时，回溯到真正的首日（否则假期第 2 天会被当成"开始"）"""
    while True:
        try:
            hol, n = get_holiday_detail(d - timedelta(days=1))
        except NotImplementedError:
            break
        if not hol or n != name:
            break
        d -= timedelta(days=1)
    return d


def upcoming_festivals(days: int = 14, today: Optional[date] = None) -> List[Tuple[date, str, int, int]]:
    """未来 N 天内的放假日素材：[(首日, 中文名, 假期天数, 距今天数)]，按首日升序。

    只报放假日、不报调休上班日；窗口内没有就空表；数据越界静默截断。
    """
    today = today or date.today()
    out: List[Tuple[date, str, int, int]] = []
    seen = set()
    d = today
    limit = today + timedelta(days=days)
    while d <= limit:
        try:
            hol, name = get_holiday_detail(d)
        except NotImplementedError:
            break                     # 数据只到已公布年份
        if hol and name in _NAME_CN and name not in seen:
            seen.add(name)
            first = _block_start(d, name)          # 假期进行中也要报真首日
            span = _span_days(first, name)
            out.append((first, _cn(name), span, (first - today).days))
            d = first + timedelta(days=span)       # 跳过整个假期块，窗口按自然日算
            continue
        d += timedelta(days=1)
    return out


def describe_next(today: Optional[date] = None) -> str:
    """对话用：下一个法定假期（不限 14 天窗口，扫到数据尽头）"""
    today = today or date.today()
    d = today
    while True:
        try:
            hol, name = get_holiday_detail(d)
        except NotImplementedError:
            return "节假日数据只覆盖到国务院已公布的年份，更远的日期查不了。"
        if hol and name in _NAME_CN:
            first = _block_start(d, name)
            span = _span_days(first, name)
            gap = (first - today).days
            if gap <= 0:  # 假期正在进行
                return (f"现在正值{_cn(name)}假期：{first.isoformat()}（周{_WEEKDAY[first.weekday()]}）"
                        f"开始，共 {span} 天，到 {(first + timedelta(days=span - 1)).isoformat()}。")
            when = "就是明天" if gap == 1 else f"还有 {gap} 天"
            return (f"下一个法定假期是{_cn(name)}：{first.isoformat()}（周{_WEEKDAY[first.weekday()]}）"
                    f"开始，放 {span} 天，{when}。")
        d += timedelta(days=1)


def _match_festival(keyword: str) -> Optional[str]:
    """口语关键词 → 包内英文名（双向包含：'端午节'含'端午'、'国庆节'含'国庆'）"""
    k = keyword.strip()
    for alias, en in _ALIAS.items():
        if alias in k or k in alias:
            return en
    return None


def describe_festival(keyword: str, today: Optional[date] = None) -> str:
    """对话用：按名称查某个法定节日的放假安排

    先从今天向后扫该节日（正在放也报真首日）；数据尽头没扫到 → 回扫今年
    报"已过 + 明年未公布"。LLM 拿到的是带起止日期和天数的完整素材，
    够它做"节前请三天假"这类日期换算。
    """
    en = _match_festival(keyword)
    if en is None:
        return (f"没认出节日'{keyword}'，能查的法定节日："
                f"元旦、春节、清明、劳动节（五一）、端午、中秋、国庆（十一）。")
    today = today or date.today()
    cn = _cn(en)

    # 向前扫：今天起最近的该节日假期块
    d = today
    while True:
        try:
            hol, name = get_holiday_detail(d)
        except NotImplementedError:
            break                      # 数据尽头，转去回看今年
        if hol and name == en:
            first = _block_start(d, name)
            span = _span_days(first, name)
            gap = (first - today).days
            if gap <= 0:
                return (f"现在正值{cn}假期：{first.isoformat()}（周{_WEEKDAY[first.weekday()]}）"
                        f"开始，共 {span} 天，到 {(first + timedelta(days=span - 1)).isoformat()}。")
            when = "就是明天" if gap == 1 else f"还有 {gap} 天"
            return (f"{cn}：{first.isoformat()}（周{_WEEKDAY[first.weekday()]}）开始，"
                    f"放 {span} 天，到 {(first + timedelta(days=span - 1)).isoformat()}，{when}。")
        d += timedelta(days=1)

    # 向后扫：今年的已经过了
    d = today - timedelta(days=1)
    while d.year == today.year:
        try:
            hol, name = get_holiday_detail(d)
        except NotImplementedError:
            break                      # 数据不覆盖年初（罕见，如实兜底）
        if hol and name == en:
            first = _block_start(d, name)
            span = _span_days(first, name)
            return (f"今年的{cn}已经过了：{first.isoformat()}（周{_WEEKDAY[first.weekday()]}）"
                    f"开始放了 {span} 天。明年的安排还没公布，公布后才能查。")
        d -= timedelta(days=1)
    return f"数据覆盖范围内没找到{cn}的放假安排（数据只到已公布年份）。"


def describe_date(d: date) -> str:
    """对话用：某一天是放假/调休上班/普通日子"""
    try:
        hol, name = get_holiday_detail(d)
    except NotImplementedError:
        return f"{d.isoformat()} 超出了节假日数据覆盖范围（只到已公布年份）。"
    wd = f"周{_WEEKDAY[d.weekday()]}"
    if hol:
        if name:                                   # 法定假日（库把普通周末也算 hol=True，要区分）
            return f"{d.isoformat()} {wd} 放假，{_cn(name)}假期。"
        if d.weekday() >= 5:
            return f"{d.isoformat()} {wd}，正常周末。"
        return f"{d.isoformat()} {wd} 放假。"      # 无名的非周末放假日（罕见，如实报）
    if name:
        return f"{d.isoformat()} {wd} 调休上班（补{_cn(name)}的假），别记成休息日哦。"
    return f"{d.isoformat()} {wd}，普通{'周末' if d.weekday() >= 5 else '工作日'}。"
