# -*- coding: utf-8 -*-
"""股票行情监控小窗口 —— 纯逻辑层。

不依赖 tkinter、不依赖任何第三方库。负责：配置加载与校验、行情源解析、
模拟行情、交易时段判定、1 分钟快照、涨幅计算、告警判定与去重状态机、
当日涨幅「突破前确认」门控（上穿前需连续 N 秒均在阈值下方），
以及独立的「1 分钟 K 线 MACD 金叉/死叉」指标状态链路。

命令行 `python core.py --selftest` 运行纯逻辑自检（不联网、不开窗口）。
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

# 北京时间固定 UTC+8，避免 Windows 缺少 tzdata 时的时区依赖
BJ = timezone(timedelta(hours=8))
LOGGER = logging.getLogger("stock_monitor")

# ---------------------------------------------------------------------------
# 默认配置（仅作兜底，不是运行时配置）
# 程序启动时优先读取同目录的 config.json；只有 config.json 缺失或 JSON 损坏
# 时才会使用下面这组值。要修改监控的股票，请改 config.json 并重启窗口。
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "stocks": [
        {"code": "sz300304", "name": "云意电气"},
        {"code": "sh601058", "name": "赛轮轮胎"},
        {"code": "sz300050", "name": "世纪鼎利"},
        {"code": "sh603893", "name": "瑞芯微"},
    ],
    "data_source": "tencent",
    "fallback_source": "sina",
    # 非交易时段（盘前/集合竞价/午休/盘后/周末）完全不联网：价格与分钟线都不拉。
    # 设为 false 可恢复旧行为（盘后仍轮询，便于复盘，但会持续发 HTTP）。
    "fetch_only_in_session": True,
    "refresh_interval_sec": 5,
    "request_timeout_sec": 3,
    "stale_threshold_sec": 20,
    "quote_max_age_sec": 90,
    "day_change_threshold_pct": 2.0,
    # 当日涨幅必须「从阈值下方突破」才提醒：上穿前需连续 N 秒所有本地观测
    # 均 <= 阈值（默认 300 秒 = 前 5 分钟）。设 0 关闭，回到「上穿即提醒」。
    "day_change_precheck_sec": 300,
    "min1_change_threshold_pct": 1.0,
    "snapshot_window_sec": 60,
    "always_on_top": True,
    "window_geometry": "380x560+120+120",
    "alert_auto_close_sec": 12,
    "alert_max_visible": 3,
    "alert_max_queue": 20,
    "mock": {
        "seed": 20261109,
        "scenario": "both",
        "prev_close": 100.0,
        "steps": [100, 100, 100, 100, 100, 100, 103.0, 100.5, 102.5, 100, 100],
    },
    # 日内 1 分钟 MACD 指标状态提示（文字信号，不绘制图表）。
    # 改动请改 config.json 的 macd 块并重启窗口；这里是配置损坏时的兜底。
    "macd": {
        "enabled": True,
        "ema_fast": 12,
        "ema_slow": 26,
        "ema_signal": 9,
        "bar_factor": 2.0,
        # prev_day = 跨交易日连续递推 EMA，当日首根继承前日末态，
        # 与同花顺 MACDFS / 通达信分时 MACD 一样开盘即有值，不再等当日 35 根。
        "warmup": "prev_day",
        "warmup_bars": 35,          # 递推序列最少根数（跨日累计），不再是当日根数
        "datalen": 1200,
        "session_windows": [["09:31", "11:30"], ["13:01", "15:00"]],
        "min_dif_ratio": 0.0001,    # |DIF| 需 ≥ 现价×该比例，只剔除几乎贴 0 的黏合交叉
        "zero_axis_filter": True,
        "bar_confirm_bars": 2,
        # 冷却 / 每日上限默认「上午无效、下午开始生效」：
        # throttle_start 之前一条都不限，到点后才施加同股冷却与每日上限。
        # 设为 "09:30" 等价于全天生效（旧行为）。
        "throttle_start": "13:00",
        # 上午已发信号是否占用下午的 daily_max 额度。
        # false（默认）= 下午重新获得 daily_max 条额度；true = 按全天累计执行上限。
        "count_morning_in_daily_max": False,
        "cooldown_min": 15,
        "daily_max": 3,
        "primary_source": "sina_kline",
        "fallback_source": "eastmoney_trends2",
        "fallback_enabled": True,
        "timeout_sec": 4,
        "retry": 1,
        "poll_delay_sec": 4,
        "per_symbol_gap_sec": 0.4,
        "max_frame_age_sec": 120,
        "low_freq_min": 3,
        "eastmoney_ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "eastmoney_ndays": 5,
        "mock_scenario": "golden",
        "mock_step_sec": 2,
        "show_badge": True,
    },
    "log_level": "INFO",
    "log_to_file": False,
}

# 内置模拟行情序列：只有列表里的第一只股票走序列，其余保持昨收横盘。
# 这三个序列只用于验证告警规则，不模拟真实价格走势：
#   day_only     —— 先在阈值下方横盘数轮，再跳到 +3%；1 分钟快照尚未形成，
#                    只触发「当日涨幅」一条（前置观察窗口需临时调小才能加速验证）
#   min1_recover —— 价格始终在昨收附近，只触发「近1分钟涨幅」，并呈现
#                    触发 → 不重复 → 回落到阈值以下解锁 → 再次上穿重触发
#   both         —— 横盘数轮后一次跳到 +3%，当日与近1分钟两条件同轮首次满足，
#                    合并为一条；回落解锁后再同轮满足，再合并一条
# 注意：day_change_precheck_sec 默认 300 秒时，这些短序列不足以自然满足
# 「前 5 分钟均在阈值下方」，验收时需把该值临时调小（见 README 验收章节）。
MOCK_SCENARIOS = {
    "day_only": {
        "prev_close": 100.0,
        "steps": [100.0, 100.0, 100.0] + [103.0] * 9,
    },
    "min1_recover": {
        "prev_close": 100.0,
        "steps": [98, 98, 98, 98, 98, 99.5, 99.6, 98.4, 101.0, 99.0, 99.0],
    },
    "both": {
        "prev_close": 100.0,
        "steps": [100, 100, 100, 100, 100, 100, 103.0, 100.5, 102.5, 100, 100],
    },
}

# MACD 离线验收场景：独立于上面的涨幅告警场景，不污染 mock.scenario 枚举。
# segments = [(根数, 起始价, 结束价), ...]，由 _segments_to_closes 展开成分钟收盘价，
# mock_minute_frame 按 tick 逐根推进，模拟 K 线逐根落地。序列只用于验证规则，
# 不代表任何真实走势。
# 说明：mock 帧只生成当日 K 线（没有跨日历史），所以不论 warmup 取哪一档，
# 参与递推的序列长度都等于当日根数，前 35 根仍是预热期；因此所有场景都以
# 「横盘 45 根」开头，之后接单一趋势，第一次方向变化即产生第一个交叉。
MOCK_MINUTE_SCENARIOS = {
    "golden":             {"session": "盘中", "segments": [(45, 100, 100), (24, 100, 103.5), (12, 103.5, 103.5)]},
    "death":              {"session": "盘中", "segments": [(45, 100, 100), (24, 100, 96.5), (12, 96.5, 96.5)]},
    "zero_axis_reject":   {"session": "盘中", "segments": [(45, 100, 100), (20, 100, 97), (15, 97, 98.3), (15, 98.3, 98.3)]},
    "bar_confirm_reject": {"session": "盘中", "segments": [(45, 100, 100), (2, 100, 103), (2, 103, 100), (20, 100, 100)]},
    "cooldown":           {"session": "盘中", "segments": [(45, 100, 100), (10, 100, 103), (4, 103, 99), (10, 99, 103), (10, 103, 103)]},
    "daily_cap":          {"session": "盘中", "segments": [(45, 100, 100), (14, 100, 103), (10, 103, 99), (14, 99, 103), (10, 103, 103)] * 4},
    "same_minute_dedup":  {"session": "盘中", "segments": [(45, 100, 100), (24, 100, 103.5), (12, 103.5, 103.5)]},
    "off_session":        {"session": "午休", "segments": [(45, 100, 100), (24, 100, 103.5), (12, 103.5, 103.5)]},
    "warmup":             {"session": "盘中", "segments": [(10, 100, 100)]},
    "fallback_switch":    {"session": "盘中", "segments": [(45, 100, 100), (24, 100, 103.5), (12, 103.5, 103.5)], "switch_at": 70},
    "null_and_dirty":     {"session": "盘中", "segments": [(8, 8.17, 8.17)]},  # 停牌全平序列
}

# 指标状态提示的固定文案：只描述指标状态，不含任何买卖动作词。
MACD_DISCLAIMER = "指标状态提示 · 非投资咨询 · 数据可能延迟、指标具滞后性 · 不构成投资建议"
MACD_METHOD_NOTE = "口径：已收盘 1 分钟 K 线收盘价 · EMA(12,26,9) 跨日连续递推 · 柱=2×(DIF−DEA)"
MACD_TPLUS1_NOTE = "A 股 T+1：当日新增持仓次一交易日方可流转；日内回转需隔夜底仓（仅机制说明）"
BADGE_GOLDEN = "MACD金叉"
BADGE_DEATH = "MACD死叉"
MACD_FORBIDDEN_WORDS = ("买入", "卖出", "加仓", "减仓", "买点", "卖点",
                        "正T", "反T", "建议买", "建议卖", "该买", "该卖")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) stock-monitor/1.0"

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawQuote:
    symbol: str
    name: str
    code: str
    price: float
    prev_close: float
    open_price: float
    volume: float
    quote_ts: datetime          # 行情时间戳（北京时间，aware）
    builtin_day_pct: Optional[float] = None


@dataclass
class Batch:
    fetch_ts: datetime          # 本地完成取数的时刻（北京时间）
    source: str
    quotes: dict = field(default_factory=dict)      # symbol -> RawQuote
    errors: dict = field(default_factory=dict)      # symbol -> 原因；"__all__" 表示整批失败
    used_fallback: bool = False


@dataclass(frozen=True)
class Alert:
    symbol: str
    name: str
    price: Optional[float]
    day_pct: Optional[float]
    min1_pct: Optional[float]
    conditions: str             # 合并后的触发条件文本
    triggered_at: datetime      # 触发时间（北京时间）


# --- 以下为独立的「1 分钟 MACD 指标状态」数据载体，不改动上面的快照链路 ---

@dataclass(frozen=True)
class MinuteBar:
    day: str                    # 交易日 YYYY-MM-DD
    minute: str                 # 已收盘 K 线标签 HHMM（4 位）
    close: float


@dataclass(frozen=True)
class MacdPoint:
    day: str
    minute: str
    close: float
    dif: float
    dea: float
    bar: float


@dataclass(frozen=True)
class MacdCross:
    symbol: str
    trade_date: str
    minute: str
    direction: str              # "golden" / "death"
    dif: float
    dea: float
    bar: float
    kline_close: float
    warmup_mode: str


@dataclass(frozen=True)
class MacdSymbolFrame:
    symbol: str
    ok: bool
    source: str
    trade_date: Optional[str]
    bars: tuple = ()            # tuple[MinuteBar]，保持不可变以便跨线程投递
    last_minute: Optional[str] = None
    used_fallback: bool = False
    error: str = ""


@dataclass(frozen=True)
class MacdFrame:
    fetch_ts: datetime
    source: str
    per_symbol: dict = field(default_factory=dict)   # symbol -> MacdSymbolFrame
    session: str = ""                               # mock 场景显式时段标记；空=按真实时钟判定


@dataclass(frozen=True)
class SignalView:
    """统一提醒视图：涨幅告警与 MACD 状态提示共用一条队列与一份弹窗布局。"""
    kind: str                   # "threshold" / "macd"
    symbol: str
    name: str
    price: Optional[float]
    day_pct: Optional[float]
    min1_pct: Optional[float]
    conditions: str
    triggered_at: datetime
    direction: str = ""
    dif: Optional[float] = None
    dea: Optional[float] = None
    bar: Optional[float] = None
    kline_minute: str = ""
    confirm_note: str = ""
    context: str = ""


@dataclass
class MacdSymbolRuntime:
    """每只股票的 MACD 计算态，仅由主线程访问。"""
    source: str = ""
    pending: Optional[MacdCross] = None
    confirm_cnt: int = 0
    prev_bar: float = 0.0
    last_minute: Optional[str] = None
    warmup_ok: bool = False
    badge: str = ""


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def normalize_symbol(raw: str) -> str:
    """把配置里的代码规范化为 sh/sz/bj + 6 位数字。"""
    s = str(raw).strip().lower()
    if re.fullmatch(r"(sh|sz|bj)\d{6}", s):
        return s
    if not re.fullmatch(r"\d{6}", s):
        raise ValueError(f"代码格式应为 sh/sz/bj+6位数字 或 6位数字: {raw!r}")
    # 前缀推断：北交所 92 号段优先于上交所 9 号段
    if s.startswith(("92", "43", "83", "87", "88")):
        prefix = "bj"
    elif s.startswith(("60", "68", "9", "5")):
        prefix = "sh"
    elif s.startswith(("00", "30", "1", "3")):
        prefix = "sz"
    else:
        raise ValueError(f"无法推断代码前缀，请显式写 sh/sz/bj: {raw!r}")
    LOGGER.info("代码 %s 未带市场前缀，推断为 %s%s", s, prefix, s)
    return prefix + s


def validate_config(cfg: dict) -> None:
    """校验配置；非法一律抛 ValueError（由 load_config 转成退出码 2）。"""
    def bad(key: str, msg: str):
        raise ValueError(f"配置项 {key} 非法: {msg}")

    stocks = cfg.get("stocks")
    if not isinstance(stocks, list) or not 3 <= len(stocks) <= 5:
        bad("stocks", "必须是 3~5 只股票的数组")
    seen = set()
    for i, item in enumerate(stocks):
        if not isinstance(item, dict):
            bad(f"stocks[{i}]", "必须是对象")
        raw = item.get("code")
        if not isinstance(raw, str) or not raw.strip():
            bad(f"stocks[{i}].code", "必填字符串")
        try:
            sym = normalize_symbol(raw)
        except ValueError as exc:
            bad(f"stocks[{i}].code", str(exc))
        if sym in seen:
            bad("stocks", f"代码重复: {sym}")
        seen.add(sym)
        if not isinstance(item.get("name", ""), str):
            bad(f"stocks[{i}].name", "必须是字符串")
        item["symbol"] = sym       # 规范化结果写回，供 app.py 使用

    def num(key: str, lo: float, hi: float, integer: bool = False):
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad(key, "必须是数字")
        if integer and not isinstance(value, int):
            bad(key, "必须是整数")
        if not lo <= value <= hi:
            bad(key, f"取值范围 [{lo}, {hi}]，当前 {value}")

    def boolean(key: str):
        if not isinstance(cfg.get(key), bool):
            bad(key, "必须是 true/false")

    if cfg.get("data_source") not in ("tencent", "sina", "mock"):
        bad("data_source", "只能是 tencent / sina / mock")
    if cfg.get("fallback_source") not in ("tencent", "sina", "mock", None):
        bad("fallback_source", "只能是 tencent / sina / mock / null")

    num("refresh_interval_sec", 1, 60, integer=True)
    num("request_timeout_sec", 1, 10, integer=True)
    num("stale_threshold_sec", 1, 600, integer=True)
    num("quote_max_age_sec", 30, 600, integer=True)
    num("day_change_threshold_pct", 0.0001, 100)
    num("day_change_precheck_sec", 0, 900, integer=True)
    num("min1_change_threshold_pct", 0.0001, 100)
    num("snapshot_window_sec", 5, 300, integer=True)
    num("alert_auto_close_sec", 0, 120, integer=True)
    num("alert_max_visible", 1, 5, integer=True)
    num("alert_max_queue", 5, 50, integer=True)
    if cfg["stale_threshold_sec"] < cfg["refresh_interval_sec"]:
        bad("stale_threshold_sec", "不能小于 refresh_interval_sec")
    if cfg["quote_max_age_sec"] < cfg["refresh_interval_sec"]:
        bad("quote_max_age_sec", "不能小于 refresh_interval_sec")

    boolean("always_on_top")
    boolean("log_to_file")
    boolean("fetch_only_in_session")
    if not re.fullmatch(r"\d{2,4}x\d{2,4}\+\d{1,4}\+\d{1,4}", str(cfg.get("window_geometry", ""))):
        bad("window_geometry", "格式应为 WxH+X+Y，例如 560x280+120+120")
    if not isinstance(cfg.get("log_level"), str):
        bad("log_level", "必须是字符串")

    mock = cfg.get("mock")
    if not isinstance(mock, dict):
        bad("mock", "必须是对象")
    if not isinstance(mock.get("seed"), int) or isinstance(mock.get("seed"), bool):
        bad("mock.seed", "必须是整数")
    if mock.get("scenario") not in (*MOCK_SCENARIOS.keys(), "custom"):
        bad("mock.scenario", "只能是 day_only / min1_recover / both / custom")
    if not isinstance(mock.get("prev_close"), (int, float)) or mock["prev_close"] <= 0:
        bad("mock.prev_close", "必须是正数")
    steps = mock.get("steps")
    if not isinstance(steps, list) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in steps
    ):
        bad("mock.steps", "必须是正数数组")
    if mock["scenario"] == "custom" and not steps:
        bad("mock.steps", "scenario=custom 时 steps 不能为空")

    validate_macd(cfg.get("macd"))


def validate_macd(macd: dict) -> None:
    """校验 macd 子对象；非法抛 ValueError，未知子键忽略（与顶层一致）。"""
    def bad(key, msg):
        raise ValueError(f"配置项 macd.{key} 非法: {msg}")

    if not isinstance(macd, dict):
        bad("", "必须是对象")

    def num(key, lo, hi, integer=False):
        value = macd.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            bad(key, "必须是数字")
        if integer and not isinstance(value, int):
            bad(key, "必须是整数")
        if not lo <= value <= hi:
            bad(key, f"取值范围 [{lo}, {hi}]，当前 {value}")

    def boolean(key):
        if not isinstance(macd.get(key), bool):
            bad(key, "必须是 true/false")

    def choice(key, options):
        if macd.get(key) not in options:
            bad(key, "只能是 " + " / ".join(map(str, options)))

    boolean("enabled")
    num("ema_fast", 2, 60, True)
    num("ema_slow", 3, 120, True)
    num("ema_signal", 2, 60, True)
    if macd["ema_slow"] <= macd["ema_fast"]:
        bad("ema_slow", "必须大于 ema_fast")
    num("bar_factor", 1, 3)
    choice("warmup", ("same_day", "prev_day"))
    num("warmup_bars", 26, 240, True)
    num("datalen", 240, 1200, True)
    num("min_dif_ratio", 0, 0.01)

    windows = macd.get("session_windows")
    if not isinstance(windows, list) or not windows:
        bad("session_windows", "必须是非空数组")
    for i, pair in enumerate(windows):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(x, str) and re.fullmatch(r"\d{2}:\d{2}", x)
                           for x in pair)):
            bad(f"session_windows[{i}]", '必须是 ["HH:MM", "HH:MM"]')
        if pair[0] >= pair[1]:
            bad(f"session_windows[{i}]", "起始必须早于结束")
        if pair[0] < "09:30" or pair[1] > "15:00":
            bad(f"session_windows[{i}]", "必须落在 09:30~15:00 内")

    boolean("zero_axis_filter")
    num("bar_confirm_bars", 0, 5, True)
    start = macd.get("throttle_start")
    if (not isinstance(start, str) or not re.fullmatch(r"\d{2}:\d{2}", start)
            or not "09:30" <= start <= "15:00"):
        bad("throttle_start", '必须是 "HH:MM" 且落在 09:30~15:00 内')
    boolean("count_morning_in_daily_max")
    num("cooldown_min", 0, 120, True)
    num("daily_max", 1, 20, True)
    choice("primary_source", ("sina_kline", "eastmoney_trends2"))
    choice("fallback_source", ("sina_kline", "eastmoney_trends2"))
    boolean("fallback_enabled")
    num("timeout_sec", 1, 10, True)
    num("retry", 0, 3, True)
    num("poll_delay_sec", 0, 30, True)
    num("per_symbol_gap_sec", 0, 3)
    num("max_frame_age_sec", 30, 600, True)
    num("low_freq_min", 1, 15, True)
    if not isinstance(macd.get("eastmoney_ut"), str) or not macd["eastmoney_ut"]:
        bad("eastmoney_ut", "必须是非空字符串")
    num("eastmoney_ndays", 1, 5, True)
    choice("mock_scenario", (*MOCK_MINUTE_SCENARIOS.keys(),))
    num("mock_step_sec", 1, 30, True)
    boolean("show_badge")


def load_config(path: str):
    """返回 (cfg, note)。配置缺失/损坏→默认配置+提示；字段非法→打印原因并退出。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    note = ""
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                user = json.load(fh)
            if not isinstance(user, dict):
                raise ValueError("顶层必须是 JSON 对象")
            # 兼容旧键名：沿用 min5_* 时不要静默忽略，否则阈值改了却不生效
            if ("min1_change_threshold_pct" not in user
                    and "min5_change_threshold_pct" in user):
                user["min1_change_threshold_pct"] = user["min5_change_threshold_pct"]
            for key, value in user.items():
                if key not in cfg:
                    continue                      # 未知键忽略
                if key == "mock" and isinstance(value, dict):
                    cfg["mock"].update(value)
                elif key == "macd" and isinstance(value, dict):
                    # 与 mock 一样浅合并：用户只写一个键时不能冲掉其余默认值
                    cfg["macd"].update(value)
                else:
                    cfg[key] = value
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            LOGGER.warning("配置文件不可用(%s)，已使用内置默认配置", exc)
            cfg = copy.deepcopy(DEFAULT_CONFIG)
            note = "配置缺失/损坏，已用默认配置"
    else:
        LOGGER.warning("配置文件不存在: %s，已使用内置默认配置", path)
        note = "配置缺失，已用默认配置"
    try:
        validate_config(cfg)
    except ValueError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        sys.exit(2)
    return cfg, note


# ---------------------------------------------------------------------------
# 时间与格式化
# ---------------------------------------------------------------------------


def beijing_now() -> datetime:
    return datetime.now(BJ)


def session_of(now_bj: datetime, quote_ts: Optional[datetime] = None) -> str:
    """按北京时间判定交易时段（不含法定节假日日历）。"""
    if now_bj.weekday() >= 5:
        return "休市"
    minute = now_bj.hour * 60 + now_bj.minute
    if minute < 9 * 60 + 15:
        return "盘前"
    if minute < 9 * 60 + 30:
        return "集合竞价"
    if minute <= 11 * 60 + 30:
        return "盘中"
    if minute < 13 * 60:
        return "午休"
    if minute <= 15 * 60:
        return "盘中"
    return "盘后"


def compute_day_pct(price: Optional[float], prev_close: Optional[float]) -> Optional[float]:
    """当日涨跌幅 = (现价 - 昨收) / 昨收 × 100%。"""
    if price is None or prev_close is None or price <= 0 or prev_close <= 0:
        return None
    return (price - prev_close) / prev_close * 100.0


def compute_min1_pct(price: Optional[float], ref_price: Optional[float]) -> Optional[float]:
    """近1分钟涨幅 = (现价 - 1分钟前快照价) / 该快照价 × 100%。"""
    if price is None or ref_price is None or price <= 0 or ref_price <= 0:
        return None
    return (price - ref_price) / ref_price * 100.0


def fmt_price(price: Optional[float]) -> str:
    if price is None:
        return "—"
    return "¥" + f"{price:,.3f}".rstrip("0").rstrip(".")


def fmt_pct(value: Optional[float], placeholder: str = "—") -> str:
    if value is None:
        return placeholder
    return f"{value:+.2f}%"


# ---------------------------------------------------------------------------
# 行情源解析
# ---------------------------------------------------------------------------

TENCENT_RE = re.compile(r'v_([a-z]{2}\d{6})="([^"]*)"')
SINA_RE = re.compile(r'var hq_str_([a-z]{2}\d{6})="([^"]*)"')


def decode_body(raw: bytes) -> str:
    """行情接口为 GBK；gb18030 是 GBK 超集，可兼容生僻字。"""
    for enc in ("gb18030", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("gb18030", errors="replace")


def _validate_quote(price: float, prev: float, open_price: float, volume: float) -> Optional[str]:
    if price <= 0 or prev <= 0:
        return "无有效报价(停牌/未开盘)"
    # 停牌/未开盘典型形态：现价=昨收、今开=0、成交量为 0
    if price == prev and open_price == 0 and volume == 0:
        return "停牌/未开盘"
    return None


def parse_tencent(text: str, symbols: list):
    """腾讯 ~ 分隔：f1名称 f2代码 f3现价 f4昨收 f5今开 f6量 f30时间戳 f32涨跌幅。

    字段索引来自 2026-11-09 实测响应；此处仍做长度/类型/时间戳/算术四重校验，
    任何异常只影响单只股票，不拖垮整批。
    """
    wanted = set(symbols)
    quotes, errors = {}, {}
    for match in TENCENT_RE.finditer(text):
        sym, payload = match.group(1), match.group(2)
        if sym not in wanted:
            continue
        f = payload.split("~")
        if len(f) < 33:
            errors[sym] = f"字段不足({len(f)})"
            continue
        if not f[1] or not re.fullmatch(r"\d{6}", f[2] or ""):
            errors[sym] = "名称/代码异常"
            continue
        try:
            price = float(f[3])
            prev = float(f[4])
            open_price = float(f[5] or 0)
            volume = float(f[6] or 0)
            builtin = float(f[32] or 0)
            quote_ts = datetime.strptime(f[30], "%Y%m%d%H%M%S").replace(tzinfo=BJ)
        except (ValueError, TypeError) as exc:
            errors[sym] = f"字段解析异常({exc})"
            continue
        reason = _validate_quote(price, prev, open_price, volume)
        if reason:
            errors[sym] = reason
            continue
        computed = compute_day_pct(price, prev)
        if builtin and abs(computed - builtin) > 0.05:
            LOGGER.warning("%s 自算涨跌幅 %.2f%% 与接口 %.2f%% 不一致", sym, computed, builtin)
        quotes[sym] = RawQuote(sym, f[1], f[2], price, prev, open_price,
                               volume, quote_ts, builtin)
    for sym in symbols:
        if sym not in quotes and sym not in errors:
            errors[sym] = "无数据(响应中缺失)"
    return quotes, errors


def parse_sina(text: str, symbols: list):
    """新浪逗号分隔：f0名称 f1今开 f2昨收 f3现价 f8量 f30日期 f31时间。"""
    wanted = set(symbols)
    quotes, errors = {}, {}
    for match in SINA_RE.finditer(text):
        sym, payload = match.group(1), match.group(2)
        if sym not in wanted:
            continue
        f = payload.split(",")
        if len(f) < 32:
            errors[sym] = f"字段不足({len(f)})"
            continue
        if not f[0]:
            errors[sym] = "名称异常"
            continue
        try:
            open_price = float(f[1])
            prev = float(f[2])
            price = float(f[3])
            volume = float(f[8] or 0)
            quote_ts = datetime.strptime(
                f"{f[30].strip()} {f[31].strip()}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=BJ)
        except (ValueError, TypeError) as exc:
            errors[sym] = f"字段解析异常({exc})"
            continue
        reason = _validate_quote(price, prev, open_price, volume)
        if reason:
            errors[sym] = reason
            continue
        quotes[sym] = RawQuote(sym, f[0], sym[2:], price, prev, open_price,
                               volume, quote_ts, None)
    for sym in symbols:
        if sym not in quotes and sym not in errors:
            errors[sym] = "无数据(响应中缺失)"
    return quotes, errors


def _is_cert_error(exc: Exception) -> bool:
    return (isinstance(exc, ssl.SSLError)
            or "SSL" in type(exc).__name__
            or "certificate" in str(exc).lower())


def _http_get(url: str, headers: dict, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _fetch_text(source: str, symbols: list, cfg: dict) -> str:
    timeout = cfg["request_timeout_sec"]
    joined = ",".join(symbols)
    if source == "tencent":
        url = "https://qt.gtimg.cn/q=" + joined
        headers = {"User-Agent": UA}
    elif source == "sina":
        url = "https://hq.sinajs.cn/list=" + joined
        headers = {"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"}
    else:
        raise ValueError(f"不支持的行情源: {source}")
    try:
        raw = _http_get(url, headers, timeout)
    except Exception as exc:
        if not _is_cert_error(exc):
            raise
        LOGGER.warning("HTTPS 证书异常，改用 HTTP 重试: %s", exc)
        raw = _http_get(url.replace("https://", "http://", 1), headers, timeout)
    return decode_body(raw)


def mock_quotes(symbols: list, cfg: dict, tick: int) -> Batch:
    """确定性模拟行情，不联网。仅第一只股票走价格序列，其余横盘。"""
    mock = cfg["mock"]
    scenario = mock["scenario"]
    if scenario == "custom":
        steps = list(mock["steps"])
        prev_close = float(mock["prev_close"])
    else:
        spec = MOCK_SCENARIOS[scenario]
        steps = list(spec["steps"])
        prev_close = float(spec["prev_close"])
    if not steps:
        raise ValueError("模拟行情序列为空")
    index = min(max(tick - 1, 0), len(steps) - 1)
    active_price = float(steps[index])
    now = beijing_now()
    # 正常运行时 cfg 已由 load_config 校验并写入 symbol；此处兼容直接构造的配置
    names = {}
    for item in cfg.get("stocks", []):
        try:
            sym = item.get("symbol") or normalize_symbol(item.get("code", ""))
        except ValueError:
            continue
        names[sym] = item.get("name") or sym
    quotes = {}
    for i, sym in enumerate(symbols):
        price = active_price if i == 0 else prev_close
        quotes[sym] = RawQuote(sym, names.get(sym, sym), sym[2:], price,
                               prev_close, prev_close, 1.0, now, None)
    return Batch(now, "mock", quotes, {})


def fetch_quotes(symbols: list, cfg: dict, tick: int = 0) -> Batch:
    """一轮取数：主源失败（或无有效报价）才切备用源；一轮只用一种源，绝不混算。"""
    source = cfg["data_source"]
    if source == "mock":
        return mock_quotes(symbols, cfg, tick)
    sources = [source]
    fallback = cfg.get("fallback_source")
    if fallback in ("tencent", "sina") and fallback != source:
        sources.append(fallback)

    failures = []
    for i, src in enumerate(sources):
        try:
            text = _fetch_text(src, symbols, cfg)
            quotes, errors = (parse_tencent if src == "tencent" else parse_sina)(text, symbols)
            if not quotes and i + 1 < len(sources):
                failures.append(f"{src}: 响应无有效报价")
                continue
            return Batch(beijing_now(), src, quotes, errors, used_fallback=(i > 0))
        except Exception as exc:
            failures.append(f"{src}: {type(exc).__name__}: {exc}")
            LOGGER.warning("行情源 %s 取数失败: %s", src, exc)
    return Batch(beijing_now(), sources[-1], {},
                 {"__all__": " | ".join(failures)},
                 used_fallback=len(sources) > 1)


# ---------------------------------------------------------------------------
# 1 分钟价格快照
# ---------------------------------------------------------------------------


class SnapshotStore:
    """每股一个带本地观测时间戳的滚动快照 deque。

    P_ref = 时间戳 <= (now - window_sec) 的最近一条有效价格；取不到即「计算中」。
    """

    def __init__(self, window_sec: int, interval_sec: int):
        self.window = window_sec
        self.keep = window_sec + 2 * interval_sec
        self._data: dict[str, deque] = {}
        self._last_prev: dict[str, float] = {}
        self._last_trade_date: dict[str, object] = {}

    def reset(self, symbol: str) -> None:
        self._data.pop(symbol, None)
        self._last_prev.pop(symbol, None)

    def append(self, symbol: str, quote: RawQuote, now: datetime) -> Optional[str]:
        """追加快照；返回重置原因（跨日 / 除权）或 None。"""
        trade_date = quote.quote_ts.astimezone(BJ).date()
        reason = None
        last_date = self._last_trade_date.get(symbol)
        if last_date is not None and last_date != trade_date:
            self.reset(symbol)
            reason = "跨日数据，快照已重置"
        self._last_trade_date[symbol] = trade_date

        last_prev = self._last_prev.get(symbol)
        if (last_prev is not None and last_prev > 0
                and abs(quote.prev_close - last_prev) / last_prev > 0.01):
            self.reset(symbol)
            reason = "昨收突变(除权/修正)，快照已重置"
        self._last_prev[symbol] = quote.prev_close

        queue_ = self._data.setdefault(symbol, deque())
        # 跨北京时间 00:00 的隔夜挂机清理
        if queue_ and datetime.fromtimestamp(queue_[0][0], BJ).date() != now.astimezone(BJ).date():
            queue_.clear()
        queue_.append((now.timestamp(), quote.price, trade_date, quote.prev_close))
        cutoff = now.timestamp() - self.keep
        while queue_ and queue_[0][0] < cutoff:
            queue_.popleft()
        return reason

    def ref_price(self, symbol: str, now: datetime) -> Optional[float]:
        cutoff = now.timestamp() - self.window
        queue_ = self._data.get(symbol)
        if not queue_:
            return None
        for ts, price, *_ in reversed(queue_):
            if ts <= cutoff:
                return price if price > 0 else None
        return None


# ---------------------------------------------------------------------------
# 告警判定与去重
# ---------------------------------------------------------------------------

CONDITION_LABEL = {"day": "当日涨幅", "min1": "近1分钟涨幅"}


class DayChangeGate:
    """当日涨幅「突破前确认」：上穿前必须已连续 N 秒都待在阈值下方。

    - 观测来自工具自身每轮价格快照（与近1分钟涨幅同源），不是行情分钟线。
    - 「下方」= 观测值 <= 阈值；== 阈值算下方，与触发用严格 > 保持互补。
    - 下列任一情况都会让连续计时重新开始：出现 > 阈值的观测、观测值为
      None（计算中/无数据）、相邻两轮间隔超过 max_gap_sec（取数中断、
      非盘中暂停取数、午休）。
    - 启动时没有历史，因此启动后必须先累积 precheck_sec 秒的下方观测。
      程序启动时就已站在阈值上方的股票，直到回落并重新满足前不会提醒。
    - 只回答「是否允许触发」，不参与去重锁。被抑制时不上锁、不消耗机会。
    """

    def __init__(self, precheck_sec: int = 0, max_gap_sec: float = 15.0):
        self.sec = int(precheck_sec)
        self.max_gap = float(max_gap_sec)
        self._below_since: dict[str, float] = {}
        self._last_obs: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.sec > 0

    def reset(self, symbol: str) -> None:
        self._below_since.pop(symbol, None)
        self._last_obs.pop(symbol, None)

    def remaining(self, symbol: str, now: datetime) -> float:
        """距离允许触发还差多少秒；0 = 已允许（或功能关闭）。"""
        if not self.enabled:
            return 0.0
        ts = now.timestamp()
        since = self._below_since.get(symbol)
        if since is None:
            return float(self.sec)
        if ts - self._last_obs.get(symbol, ts) > self.max_gap:
            return float(self.sec)
        return max(0.0, self.sec - (ts - since))

    def observe(self, symbol: str, now: datetime, day_pct: Optional[float],
                threshold: float) -> None:
        """在本轮判定之后记录观测；判定永远先于本轮记录，本轮点不算「前 N 秒」。"""
        if not self.enabled:
            return
        ts = now.timestamp()
        last = self._last_obs.get(symbol)
        gap_ok = last is None or ts - last <= self.max_gap
        if not gap_ok or day_pct is None or day_pct > threshold:
            # 中断 / 数据缺失 / 又站上阈值上方 → 连续计时清零，等待下一根下方观测
            self._below_since.pop(symbol, None)
        elif symbol not in self._below_since:
            self._below_since[symbol] = ts       # 第一根下方观测开始计时
        # 否则保持原起点，连续下方时长继续累积
        self._last_obs[symbol] = ts


class AlertEngine:
    """每个 (股票, 条件) 独立上锁。

    - 触发：严格 > 阈值且未上锁 → 上锁并记录本次触发（不重复提醒）。
    - 恢复：严格 < 阈值 → 解锁；之后再次上穿才重新提醒。
    - == 阈值：视为仍 >= 阈值，保持锁状态。
    - 指标为 None（计算中/无数据/停牌）：冻结锁状态，不判定、不误解锁。
    - 同一轮同一股票新触发的多个条件合并为一条提醒。
    - day 条件额外受 DayChangeGate 门控：上穿前需连续 precheck_sec 秒所有
      观测均 <= 阈值。门控不通过时**不上锁、不提醒、不消耗触发机会**，
      回落并重新满足条件后仍可触发；min1 条件完全不受门控影响。
    """

    def __init__(self, day_threshold: float, min1_threshold: float,
                 day_precheck_sec: int = 0, max_gap_sec: float = 15.0):
        self.day_threshold = day_threshold
        self.min1_threshold = min1_threshold
        self.locked: dict[tuple[str, str], bool] = {}
        self.day_gate = DayChangeGate(day_precheck_sec, max_gap_sec)

    def reset_symbol(self, symbol: str) -> None:
        for cond in ("day", "min1"):
            self.locked[(symbol, cond)] = False
        self.day_gate.reset(symbol)     # 跨日 / 除权：旧观测不能充当前置历史

    def day_precheck_remaining(self, symbol: str, now: datetime) -> float:
        """供界面展示「还需确认多久」；必须在 evaluate 之前取值。"""
        return self.day_gate.remaining(symbol, now)

    def evaluate(self, symbol: str, name: str, price: Optional[float],
                 day_pct: Optional[float], min1_pct: Optional[float],
                 now: datetime) -> Optional[Alert]:
        triggers = []
        day_armed = self.day_gate.remaining(symbol, now) <= 0
        for cond, value, threshold in (
            ("day", day_pct, self.day_threshold),
            ("min1", min1_pct, self.min1_threshold),
        ):
            if value is None:
                continue
            key = (symbol, cond)
            is_locked = self.locked.get(key, False)
            if value > threshold:
                if not is_locked:
                    if cond == "day" and not day_armed:
                        LOGGER.info(
                            "当日涨幅 %.2f%% 上穿，但前 %ds 未满足「均在 %.2f%% 下方」，"
                            "本次抑制且不上锁（还差 %.0fs）",
                            value, self.day_gate.sec, threshold,
                            self.day_gate.remaining(symbol, now))
                    else:
                        self.locked[key] = True
                        triggers.append(cond)
                        LOGGER.info("告警触发 %s %s = %.2f%% > %.2f%%",
                                    symbol, cond, value, threshold)
            elif value < threshold:
                if is_locked:
                    LOGGER.info("告警恢复 %s %s = %.2f%% < %.2f%%，锁已解除",
                                symbol, cond, value, threshold)
                self.locked[key] = False
        # 判定结束后才记录本轮观测，保证「前 N 秒」不含当前这个上穿点
        self.day_gate.observe(symbol, now, day_pct, self.day_threshold)
        if not triggers:
            return None
        parts = []
        for cond in triggers:
            threshold = self.day_threshold if cond == "day" else self.min1_threshold
            parts.append(f"{CONDITION_LABEL[cond]} > {threshold:.2f}%")
        return Alert(symbol, name, price, day_pct, min1_pct,
                     "；".join(parts), now)


# ---------------------------------------------------------------------------
# 1 分钟 K 线数据源与解析（新浪主源 / 东财备用源，绝不混源）
# ---------------------------------------------------------------------------

MACD_SOURCE_LABEL = {"sina_kline": "新浪分钟K", "eastmoney_trends2": "东财分时"}


def to_eastmoney_secid(symbol: str) -> str:
    """sh→1. / sz→0. / bj→0.（2026-11-09 实测 0.430047 有效、1.430047 为 null）。"""
    return ("1." if symbol.startswith("sh") else "0.") + symbol[2:]


def build_sina_kline_url(symbol: str, datalen: int) -> str:
    return ("https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
            f"?symbol={symbol}&scale=1&ma=no&datalen={datalen}")


def build_eastmoney_trends2_url(symbol: str, cfg: dict) -> str:
    macd = cfg["macd"]
    return ("https://push2his.eastmoney.com/api/qt/stock/trends2/get"
            f"?secid={to_eastmoney_secid(symbol)}&ut={macd['eastmoney_ut']}"
            "&fields1=f1,f2,f3,f4,f5,f6,f7,f8"
            "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"
            f"&ndays={macd['eastmoney_ndays']}&iscr=0&iscca=0")


def _http_get_utf8(url: str, headers: dict, timeout: int) -> str:
    """分钟线接口返回 UTF-8（新浪 header 谎称 gbk，实测按 UTF-8 正常）。"""
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _minute_bar(day: str, clock: str, price) -> Optional[MinuteBar]:
    """把「日期 + 时间 + 收盘价」规范化成一根 K 线；异常返回 None 由调用方跳过。"""
    try:
        minute = str(clock)[:5].replace(":", "")
        value = float(price)
        if (value > 0 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day))
                and re.fullmatch(r"\d{4}", minute)):
            return MinuteBar(str(day), minute, value)
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def parse_sina_kline(text: str, symbol: str) -> list:
    """新浪返回顶层数组；无效代码为字面量 null；close 是字符串。"""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"新浪分钟K线非 JSON: {exc}") from exc
    if payload is None:
        raise ValueError("新浪分钟K线返回 null（无数据/代码无效）")
    if not isinstance(payload, list):
        raise ValueError("新浪分钟K线顶层不是数组")
    bars = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            day, clock = str(row["day"]).split(" ")
        except (KeyError, ValueError):
            continue
        bar = _minute_bar(day, clock, row.get("close"))
        if bar:
            bars.append(bar)                 # 单根异常只跳过该根
    return bars


def parse_eastmoney_trends2(text: str, symbol: str) -> list:
    """东财 data.trends 是字符串数组，逗号切分 [0] 时间、[2] 收盘价。"""
    try:
        payload = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"东财分时非 JSON: {exc}") from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("trends") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("东财分时返回空数据/data:null")
    bars = []
    for row in rows:
        fields = str(row).split(",")
        if len(fields) < 3:
            continue
        try:
            day, clock = fields[0].strip().split(" ")
        except ValueError:
            continue
        bar = _minute_bar(day, clock, fields[2])
        if bar:
            bars.append(bar)
    return bars


def normalize_bars(raw: list, now_hhmm: str = "1500") -> tuple:
    """截断盘后脏点、按 (日,分钟) 去重留末根、排序；返回 (bars, trade_date, last_minute)。

    bars 保留全部历史根（供 prev_day 预热与切源整段重算），判定交叉时再按日筛选。
    """
    if not raw:
        return (), None, None
    latest = {}
    for bar in raw:
        if bar.minute > "1500":            # 腾讯分时等可能混入 1506~1530 常数点
            continue
        latest[(bar.day, bar.minute)] = bar
    bars = tuple(sorted(latest.values(), key=lambda b: (b.day, b.minute)))
    if not bars:
        return (), None, None
    trade_date = max(b.day for b in bars)
    minutes = [b.minute for b in bars if b.day == trade_date and b.minute <= now_hhmm]
    return bars, trade_date, (max(minutes) if minutes else None)


def fetch_minute_bars(symbol: str, cfg: dict, fetcher=None,
                      now_hhmm: Optional[str] = None) -> MacdSymbolFrame:
    """取单只股票分钟线。主源失败/空数据 → 整段改用备用源序列，绝不混源。

    now_hhmm 默认取当前北京时刻；自检注入固定值，避免断言随运行时刻变化。
    """
    macd = cfg["macd"]
    sources = [macd["primary_source"]]
    if macd["fallback_enabled"] and macd["fallback_source"] not in sources:
        sources.append(macd["fallback_source"])
    failures = []
    for index, src in enumerate(sources):
        url = (build_sina_kline_url(symbol, macd["datalen"])
               if src == "sina_kline" else build_eastmoney_trends2_url(symbol, cfg))
        for attempt in range(macd["retry"] + 1):
            try:
                text = (fetcher(url) if fetcher is not None
                        else _http_get_utf8(url, {"User-Agent": UA}, macd["timeout_sec"]))
                raw = (parse_sina_kline if src == "sina_kline"
                       else parse_eastmoney_trends2)(text, symbol)
                bars, trade_date, last_minute = normalize_bars(
                    raw, now_hhmm or beijing_now().strftime("%H%M"))
                if not bars or not trade_date or last_minute is None:
                    raise ValueError("当日无有效分钟线")
                today_bars = [b for b in bars if b.day == trade_date]
                # 开盘 09:31 当日只有 1 根，价格集合天然为 1，不能据此判停牌；
                # 只有「已经有多根当日 K 线却全部同价」才是停牌/全平序列。
                if len(today_bars) > 1 and len({b.close for b in today_bars}) == 1:
                    raise ValueError("停牌/无有效分钟线（全平序列）")
                return MacdSymbolFrame(symbol, True, src, trade_date, bars,
                                       last_minute, used_fallback=(index > 0))
            except Exception as exc:
                failures.append(f"{src}: {type(exc).__name__}: {exc}")
                if attempt < macd["retry"]:
                    continue
                break
        LOGGER.warning("%s 分钟线源 %s 取数失败", symbol, src)
    return MacdSymbolFrame(symbol, False, sources[0], None, (), None,
                           False, " | ".join(failures))


def fetch_macd_frame(symbols: list, cfg: dict, fetcher=None,
                     sleep_fn=None) -> MacdFrame:
    """一轮分钟线取数（串行）；sleep_fn 由界面层注入以实现可中断的限速。"""
    per_symbol = {}
    for index, sym in enumerate(symbols):
        if index and sleep_fn and cfg["macd"]["per_symbol_gap_sec"]:
            sleep_fn(cfg["macd"]["per_symbol_gap_sec"])
        per_symbol[sym] = fetch_minute_bars(sym, cfg, fetcher)
    return MacdFrame(beijing_now(), cfg["macd"]["primary_source"], per_symbol)


def macd_cursor_advance(cursor: dict, symbol: str,
                        frame: MacdSymbolFrame) -> tuple[bool, Optional[tuple]]:
    """分钟线帧去重：返回 (是否投递给主线程, 要写入的新游标)。

    游标必须含「数据源 + 是否走备用源」：主源切备用源时 trade_date 与
    last_minute 往往完全相同，若只用这两项做键，换源帧会被整帧吞掉，
    MacdSignalEngine.reset_compute() 永远不触发，旧源的柱确认候选会混进新源。
    失败帧要投递（主线程显示「分钟线失败」），但绝不能覆盖有效游标。
    """
    if not frame.ok:
        return True, None
    key = (frame.trade_date, frame.last_minute,
           frame.source, frame.used_fallback)
    prev = cursor.get(symbol)
    if prev == key:
        return False, None                       # 同一分钟同源，不重复计算
    if (prev and prev[0] == key[0] and prev[1] and key[1]
            and key[1] < prev[1]):
        return False, None                       # 时间倒退，不回退游标
    return True, key


def _segments_to_closes(segments: list) -> list:
    out = []
    for count, start, end in segments:
        for i in range(count):
            out.append(round(start + (end - start) * i / max(count - 1, 1), 4))
    return out


def _mock_minute_labels(count: int) -> list:
    """09:31 起连续分钟标签，11:30 之后跳到 13:01（与真实分钟线一致）。"""
    labels, clock = [], 9 * 60 + 31
    for _ in range(count):
        hour, minute = divmod(clock, 60)
        labels.append(f"{hour:02d}{minute:02d}")
        clock += 1
        if clock == 11 * 60 + 31:
            clock = 13 * 60 + 1
    return labels


def mock_minute_frame(symbol: str, cfg: dict, tick: int) -> MacdFrame:
    """离线可复现的分钟线帧：按 tick 逐根推进，收盘后也能验收。"""
    macd = cfg["macd"]
    scenario = macd["mock_scenario"]
    spec = MOCK_MINUTE_SCENARIOS[scenario]
    day = beijing_now().date().isoformat()
    source = macd["primary_source"]
    if scenario == "null_and_dirty":
        frame = MacdSymbolFrame(symbol, False, source, day, (), None,
                                False, "模拟：停牌/无有效分钟线")
        return MacdFrame(beijing_now(), source, {symbol: frame}, spec["session"])
    closes = _segments_to_closes(spec["segments"])
    labels = _mock_minute_labels(len(closes))
    count = min(max(tick, 1), len(closes))
    bars = tuple(MinuteBar(day, labels[i], closes[i]) for i in range(count))
    switched = bool(spec.get("switch_at") and tick > spec["switch_at"])
    if switched:
        source = macd["fallback_source"]
    frame = MacdSymbolFrame(symbol, True, source, day, bars, labels[count - 1],
                            used_fallback=switched)
    return MacdFrame(beijing_now(), source, {symbol: frame}, spec["session"])


# ---------------------------------------------------------------------------
# MACD 计算、交叉判定与离散事件状态机（与 AlertEngine 的连续阈值模型完全分离）
# ---------------------------------------------------------------------------


def compute_macd(bars, cfg: dict) -> list:
    """EMA12/EMA26 → DIF → DEA9 → 柱，adjust=False 全量递推。

    warmup=prev_day（默认）：用完整多日序列连续递推，EMA 不在交易日边界重置，
    当日首根继承前日末态。这是同花顺 MACDFS「09:30 就有值」的关键——
    首根 DIF/DEA 已经收敛，无需当日 35 根预热。
    warmup=same_day：只取当日 bars，首根 EMA12=EMA26=close、DIF=0、DEA=0，
    等价于「当日从头」，适合与只看当日序列的口径做数值对照。
    """
    macd = cfg["macd"]
    if not bars:
        return []
    today = max(b.day for b in bars)
    seq = (list(bars) if macd["warmup"] == "prev_day"
           else [b for b in bars if b.day == today])
    if not seq:
        return []
    kf = 2.0 / (macd["ema_fast"] + 1)
    ks = 2.0 / (macd["ema_slow"] + 1)
    kd = 2.0 / (macd["ema_signal"] + 1)
    ef = es = seq[0].close
    dea = 0.0
    points = []
    for index, bar in enumerate(seq):
        if index == 0:
            dif = 0.0
        else:
            ef = bar.close * kf + ef * (1 - kf)
            es = bar.close * ks + es * (1 - ks)
            dif = ef - es
            dea = dif * kd + dea * (1 - kd)
        points.append(MacdPoint(bar.day, bar.minute, bar.close, dif, dea,
                                macd["bar_factor"] * (dif - dea)))
    return points


def detect_cross(points: list, cfg: dict, symbol: str = "",
                 trade_date: Optional[str] = None) -> list:
    """只用当日「已收盘的相邻两根」判交叉；0 轴顺势过滤在此完成。"""
    macd = cfg["macd"]
    if trade_date is None:
        trade_date = max((p.day for p in points), default=None)
    if not trade_date:
        return []
    result = []
    for prev, curr in zip(points, points[1:]):
        if prev.day != trade_date or curr.day != trade_date:
            continue                                  # 跨日/午休断点不与前日末根比
        golden = prev.dif <= prev.dea and curr.dif > curr.dea
        death = prev.dif >= prev.dea and curr.dif < curr.dea
        if not (golden or death):
            continue
        if macd["zero_axis_filter"] and (
                (golden and curr.dif <= 0) or (death and curr.dif >= 0)):
            continue
        # 0 轴黏合过滤：|DIF| 相对现价过小说明两线几乎贴在一起，
        # 开盘阶段这类「擦线」交叉极多，用相对阈值避免跨价格量纲失效。
        min_dif = macd.get("min_dif_ratio", 0.0) * abs(curr.close)
        if min_dif > 0 and abs(curr.dif) < min_dif:
            continue
        result.append(MacdCross(symbol, trade_date, curr.minute,
                                "golden" if golden else "death",
                                curr.dif, curr.dea, curr.bar, curr.close,
                                macd["warmup"]))
    return result


def macd_in_session_windows(now: datetime, cfg: dict) -> bool:
    """当前北京时间是否落在允许出信号的时段窗口内。"""
    hhmm = now.astimezone(BJ).strftime("%H%M")
    return any(start.replace(":", "") <= hhmm <= end.replace(":", "")
               for start, end in cfg["macd"]["session_windows"])


def macd_throttle_active(now: datetime, cfg: dict) -> bool:
    """冷却 / 每日上限此刻是否生效。

    默认在 throttle_start（13:00）之前不生效，即上午不限流；到点后（下午）
    才施加同股冷却与每日上限。上午发出的信号既不启动冷却计时，也不占用下午
    额度，保证下午第一条信号不会被午休前的信号误挡。
    把 throttle_start 设为 "09:30" 即恢复「全天生效」的旧行为。
    """
    hhmm = now.astimezone(BJ).strftime("%H:%M")
    return hhmm >= cfg["macd"].get("throttle_start", "13:00")


def macd_http_allowed(now: datetime, cfg: dict) -> bool:
    """分钟线是否允许联网请求。

    fetch_only_in_session=true（默认）时严格只在盘中请求；
    设为 false 则盘后也允许拉取（复盘用，代价是持续发 HTTP）。
    """
    if cfg.get("fetch_only_in_session", True):
        return session_of(now.astimezone(BJ)) == "盘中"
    return True


def alert_to_view(alert: Alert) -> SignalView:
    """涨幅告警 → 统一提醒视图，字段与文案保持原样。"""
    return SignalView(kind="threshold", symbol=alert.symbol, name=alert.name,
                      price=alert.price, day_pct=alert.day_pct,
                      min1_pct=alert.min1_pct, conditions=alert.conditions,
                      triggered_at=alert.triggered_at)


def macd_cross_to_view(cross: MacdCross, state: dict, cfg: dict,
                       triggered_at: datetime, source: str = "") -> SignalView:
    """MACD 状态 → 统一提醒视图；只报指标状态，不含任何买卖动作词。"""
    macd = cfg["macd"]
    state = state or {}
    label = "金叉" if cross.direction == "golden" else "死叉"
    warmup_text = "当日从头" if macd["warmup"] == "same_day" else "多日连续"
    rate_note = (f"{macd['cooldown_min']} 分钟内同股不重复提示 · "
                 f"每只每日最多 {macd['daily_max']} 条"
                 if macd_throttle_active(triggered_at, cfg)
                 else "上午不限冷却/日上限 · 下午起按冷却与每日上限限流")
    return SignalView(
        kind="macd", symbol=cross.symbol,
        name=state.get("name") or cross.symbol,
        price=state.get("price"), day_pct=state.get("day"),
        min1_pct=state.get("min1"),
        conditions=f"1分钟 MACD {label}", triggered_at=triggered_at,
        direction=cross.direction, dif=cross.dif, dea=cross.dea, bar=cross.bar,
        kline_minute=cross.minute,
        confirm_note=(f"柱连续 {macd['bar_confirm_bars']} 根同向放大确认 · "
                      f"{rate_note}"),
        context=(f"数据源 {MACD_SOURCE_LABEL.get(source, '分钟K线')} · "
                 f"{warmup_text} EMA({macd['ema_fast']},{macd['ema_slow']},"
                 f"{macd['ema_signal']}) · 递推序列 ≥{macd['warmup_bars']} 根"),
    )


class MacdSignalEngine:
    """离散交叉事件状态机，只在主线程访问；now 显式注入便于虚拟时间验收。

    - 事件去重键 (symbol, 交易日, 交叉所在分钟, 方向) —— 全天都生效
    - 柱连续 N 根同向放大才确认（延迟约 N 分钟）—— 全天都生效
    - 冷却 / 每日上限只在 macd_throttle_active(now) 为真时生效（默认下午起）
      上午不限流，且不计入下午额度、不启动冷却计时；抑制即丢弃、不排队补发
    - 换源只清计算态，保留 _seen/_daily/_last_emit
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg["macd"]["enabled"]
        self.rt: dict[str, MacdSymbolRuntime] = {}
        self._seen: set = set()
        self._last_emit: dict[str, datetime] = {}
        self._daily: dict[tuple, int] = {}
        self._trade_date: Optional[str] = None

    def _rt(self, symbol: str) -> MacdSymbolRuntime:
        return self.rt.setdefault(symbol, MacdSymbolRuntime())

    def on_new_day(self, trade_date: str) -> bool:
        """只在交易日「前进」时清全局状态。

        停牌股或滞后的备用源偶尔会回吐前一交易日的完整序列；若允许时间倒退
        也触发清空，全局 _seen / _daily / _last_emit 会被反复清零，去重、
        冷却与每日上限全部失效。调用方应每帧只调用一次并传入最新交易日。
        """
        if not trade_date:
            return False
        if self._trade_date is not None and trade_date <= self._trade_date:
            return False
        self._trade_date = trade_date
        self._seen.clear()
        self._daily.clear()
        self._last_emit.clear()          # 冷却不跨交易日
        for rt in self.rt.values():
            rt.pending = None
            rt.confirm_cnt = 0
            rt.prev_bar = 0.0
            rt.last_minute = None
        return True

    def reset_symbol(self, symbol: str, reason: str = "") -> None:
        """跨日/除权/停牌：清计算态与当日计数，避免 EMA 断档产生伪信号。"""
        rt = self._rt(symbol)
        rt.pending = None
        rt.confirm_cnt = 0
        rt.prev_bar = 0.0
        rt.last_minute = None
        rt.badge = ""
        for key in [k for k in self._daily if k[0] == symbol]:
            self._daily.pop(key, None)
        if reason:
            LOGGER.info("%s MACD 计算态已重置: %s", symbol, reason)

    def reset_compute(self, symbol: str, new_source: str) -> None:
        """换源：丢弃旧源全部计算态，用新源完整序列从头递推。"""
        rt = self._rt(symbol)
        rt.source = new_source
        rt.pending = None
        rt.confirm_cnt = 0
        rt.prev_bar = 0.0
        rt.last_minute = None

    def _allow_emit(self, symbol: str, cross: MacdCross, now: datetime) -> bool:
        macd = self.cfg["macd"]
        rt = self._rt(symbol)
        key = (symbol, cross.trade_date, cross.minute, cross.direction)
        if key in self._seen:
            return False
        self._seen.add(key)                       # 被抑制也记录，防重放
        throttled = macd_throttle_active(now, self.cfg)
        day_key = (symbol, cross.trade_date)
        if throttled:
            last = self._last_emit.get(symbol)
            if (last is not None
                    and (now - last).total_seconds() < macd["cooldown_min"] * 60):
                rt.badge = "冷却中"
                return False
            if self._daily.get(day_key, 0) >= macd["daily_max"]:
                rt.badge = "今日已满"
                return False
        # 上午默认不占用下午额度；count_morning_in_daily_max=true 时按全天累计
        if throttled or macd.get("count_morning_in_daily_max", False):
            self._daily[day_key] = self._daily.get(day_key, 0) + 1
        if throttled:
            self._last_emit[symbol] = now        # 冷却计时从限流生效时才起算
        rt.badge = BADGE_GOLDEN if cross.direction == "golden" else BADGE_DEATH
        return True

    def process(self, symbol: str, frame: MacdSymbolFrame, points: list,
                crosses: list, now: datetime, state: Optional[dict] = None,
                session_ok: bool = True, static_ok: bool = False):
        """返回本轮应弹出的 SignalView 列表；同时更新该股角标文本。"""
        macd = self.cfg["macd"]
        rt = self._rt(symbol)
        if not self.enabled:
            rt.badge = ""
            return []
        if not frame.ok:
            rt.badge = "分钟线失败"
            return []
        if static_ok or not session_ok:
            rt.badge = "静默"
            return []

        today_points = [p for p in points if p.day == frame.trade_date]
        if not today_points:
            rt.badge = "静默"                 # 跨日序列有值但当日尚无 K 线
            return []
        # 预热门槛数的是「真正参与 EMA 递推的序列长度」：
        # prev_day 下是整个跨日序列（约 1200 根，天然达标 → 开盘即有效）；
        # same_day 下才是当日根数（需等到第 35 根 → 约 10:05）。
        warmup_points = (points if macd["warmup"] == "prev_day"
                         else today_points)
        if len(warmup_points) < macd["warmup_bars"]:
            rt.warmup_ok = False
            rt.badge = f"预热 {len(warmup_points)}/{macd['warmup_bars']}"
            return []
        rt.warmup_ok = True
        if rt.badge.startswith("预热"):
            rt.badge = ""      # 达标后必须清掉进度，否则无信号时角标永远停在 n/35

        if rt.source != frame.source:
            self.reset_compute(symbol, frame.source)
        if rt.last_minute == frame.last_minute:
            return []                                  # 同一分钟不重复计算/不重复提醒
        rt.last_minute = frame.last_minute

        emitted = []
        last = today_points[-1]
        if rt.pending is not None:
            same_side = (last.bar > 0 if rt.pending.direction == "golden"
                         else last.bar < 0)
            growing = abs(last.bar) > abs(rt.prev_bar)
            if same_side and growing:
                rt.confirm_cnt += 1
                if rt.confirm_cnt >= macd["bar_confirm_bars"]:
                    cross = rt.pending
                    if self._allow_emit(symbol, cross, now):
                        emitted.append(macd_cross_to_view(cross, state, self.cfg, now, frame.source))
                    rt.pending = None
                    rt.confirm_cnt = 0
            else:
                rt.pending = None                       # 反向或缩小 → 丢弃候选
                rt.confirm_cnt = 0

        for cross in crosses:
            if cross.minute != frame.last_minute:
                continue                                # 只认本根新产生的交叉
            if macd["bar_confirm_bars"] == 0:
                if self._allow_emit(symbol, cross, now):
                    emitted.append(macd_cross_to_view(cross, state, self.cfg, now, frame.source))
                rt.pending = None
            else:
                rt.pending = cross
            rt.confirm_cnt = 0
        rt.prev_bar = last.bar
        if rt.pending is not None and not emitted:
            rt.badge = ""                               # 确认中，角标留空不误导
        return emitted


# ---------------------------------------------------------------------------
# 纯逻辑自检（不联网、不开窗口，仅验收用）
# ---------------------------------------------------------------------------

_TENCENT_SAMPLE = (
    'v_sh600519="1~贵州茅台~600519~1448.27~1453.00~1440.06~12796~6706~6089~'
    '1448.27~3~1447.80~1~1447.31~1~1447.30~1~1447.28~2~1448.28~2~1448.38~2~'
    '1448.79~1~1448.92~2~1448.93~2~~20261109105322~-4.73~-0.33~1449.99~1435.08~'
    '1448.27/12796/1846086451~12796~184609~0.10~22.04~~1449.99~1435.08~1.03~'
    '18104.56~18104.56~6.68~1598.30~1307.70~0.59~-1~1442.75~21.06~21.99~~~'
    '-0.02~184608.6451~0.0000~0~ A~GP-A~7.35~0.64~3.59~30.29~26.74~1539.98~'
    '1151.01~7.28~12.44~10.62~1250081601~1250081601~-5.88~-5.15~1250081601~~~'
    '4.84~-0.08~~CNY~0~___D__F__N~1449.01~-17~";'
)

_SINA_SAMPLE = (
    'var hq_str_sh600519="贵州茅台,1440.060,1453.000,1442.170,1449.990,1435.080,'
    '1442.080,1442.660,1472361,2124646670.000,100,1442.080,200,1442.000,200,'
    '1441.950,500,1441.800,100,1441.570,200,1442.660,200,1442.710,300,1442.730,'
    '600,1442.740,100,1443.030,2026-11-09,11:30:00,00,";'
)


def _selftest() -> int:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")
    checks = 0

    def ck(condition, message):
        nonlocal checks
        assert condition, message
        checks += 1

    # 1. 腾讯实测样例解析
    q, e = parse_tencent(_TENCENT_SAMPLE, ["sh600519", "sz000001"])
    ck(abs(q["sh600519"].price - 1448.27) < 1e-9, "腾讯现价解析")
    ck(abs(q["sh600519"].prev_close - 1453.00) < 1e-9, "腾讯昨收解析")
    ck(q["sh600519"].quote_ts == datetime(2026, 11, 9, 10, 53, 22, tzinfo=BJ), "腾讯时间戳解析")
    ck(abs(compute_day_pct(q["sh600519"].price, q["sh600519"].prev_close)
           - q["sh600519"].builtin_day_pct) < 0.01, "自算涨跌幅与接口一致")
    ck("sz000001" in e, "批量响应缺股只影响单只")

    # 2. 新浪实测样例解析
    q2, _ = parse_sina(_SINA_SAMPLE, ["sh600519"])
    ck(abs(q2["sh600519"].price - 1442.17) < 1e-9, "新浪现价解析")
    ck(abs(q2["sh600519"].prev_close - 1453.00) < 1e-9, "新浪昨收解析")
    ck(q2["sh600519"].quote_ts == datetime(2026, 11, 9, 11, 30, 0, tzinfo=BJ), "新浪时间戳解析")

    # 3. 停牌/异常字段防御（构造字段齐全的响应，确保走到报价校验分支）
    def tencent_fields(price, prev, open_price, volume):
        parts = ["51", "平安银行", "000001", str(price), str(prev),
                 str(open_price), str(volume)]
        parts += ["0"] * 23                        # 索引 7~29
        parts += ["20261109100000", "0.07", "0.59", "11.88", "11.76"]
        return 'v_sz000001="' + "~".join(parts) + '";'

    q3, e3 = parse_tencent(tencent_fields(0, 0, 0, 0), ["sz000001"])
    ck(not q3 and "无有效报价" in e3["sz000001"], "无有效报价被拦截")
    q4, e4 = parse_tencent(tencent_fields(8.17, 8.17, 0, 0), ["sz000001"])
    ck(not q4 and "停牌" in e4["sz000001"], "停牌形态被拦截")

    # 4. 时段判定
    ck(session_of(datetime(2026, 11, 9, 10, 0, tzinfo=BJ)) == "盘中", "盘中")
    ck(session_of(datetime(2026, 11, 9, 11, 45, tzinfo=BJ)) == "午休", "午休")
    ck(session_of(datetime(2026, 11, 7, 10, 0, tzinfo=BJ)) == "休市", "周末休市")
    ck(session_of(datetime(2026, 11, 9, 15, 30, tzinfo=BJ)) == "盘后", "盘后")

    # 5. 1 分钟快照口径 / 跨日 / 除权
    store = SnapshotStore(window_sec=10, interval_sec=5)
    base = datetime(2026, 11, 9, 10, 0, 0, tzinfo=BJ)

    def rq(price, prev=100.0, ts=base):
        return RawQuote("sh600519", "测试", "600519", price, prev, prev, 1.0, ts, None)

    for offset, price in enumerate((100, 100, 100, 101.5)):
        store.append("sh600519", rq(price, ts=base), base + timedelta(seconds=offset * 5))
    now = base + timedelta(seconds=15)
    ck(store.ref_price("sh600519", now) == 100.0, "P_ref 取 <= now-10s 的最近快照")
    ck(abs(compute_min1_pct(101.5, store.ref_price("sh600519", now)) - 1.5) < 1e-9,
       "近1分钟涨幅计算")
    ck(store.ref_price("sh600519", base + timedelta(seconds=5)) is None, "快照不足时返回 None")
    ck("跨日" in (store.append("sh600519", rq(100.0, ts=base + timedelta(days=1)),
                              base + timedelta(days=1)) or ""), "跨日清快照")
    ck("昨收突变" in (store.append("sh600519", rq(100.0, prev=120.0,
                                              ts=base + timedelta(days=1)),
                                  base + timedelta(days=1)) or ""), "除权/昨收突变清快照")

    # 6. 告警：触发、去重、== 边界、恢复、重触发、合并、None 冻结
    engine = AlertEngine(2.0, 1.0)
    t0 = datetime(2026, 11, 9, 10, 0, tzinfo=BJ)
    a = engine.evaluate("s1", "测试", 102.5, 2.5, None, t0)
    ck(a is not None and "当日涨幅" in a.conditions and "近1分钟" not in a.conditions,
       "仅当日条件触发")
    ck(engine.evaluate("s1", "测试", 102.6, 2.6, None, t0) is None, "未恢复不重复提醒")
    ck(engine.evaluate("s1", "测试", 102.0, 2.0, None, t0) is None, "== 阈值保持上锁")
    ck(engine.evaluate("s1", "测试", 101.9, 1.9, None, t0) is None, "回落阈值以下解锁")
    ck(engine.evaluate("s1", "测试", 102.2, 2.2, None, t0) is not None, "再次上穿重新提醒")

    b = engine.evaluate("s2", "测试2", 103.0, 3.0, 1.5, t0)
    ck(b is not None and "当日涨幅" in b.conditions and "近1分钟涨幅" in b.conditions,
       "两条件同时满足合并为一条")
    ck(engine.evaluate("s2", "测试2", 103.1, 3.1, None, t0) is None, "day 锁 + min1 缺失不重复")
    ck(engine.evaluate("s2", "测试2", 103.2, 3.2, 1.6, t0) is None, "None 期间不误解锁")
    ck(engine.evaluate("s3", "测试3", 100.0, None, None, t0) is None, "指标缺失不判定")

    # 6b. 当日涨幅「突破前确认」：上穿前必须已连续 N 秒都在阈值下方
    gate = DayChangeGate(precheck_sec=20, max_gap_sec=15)
    g0 = datetime(2026, 11, 9, 9, 30, tzinfo=BJ)
    ck(gate.remaining("s1", g0) == 20, "无历史时前置窗口未满足（启动静默）")
    for s in (0, 5, 10, 15):
        gate.observe("s1", g0 + timedelta(seconds=s), 1.0, 2.0)
    ck(gate.remaining("s1", g0 + timedelta(seconds=20)) == 0,
       "连续下方观测满 20s 后允许触发")
    gate.observe("s1", g0 + timedelta(seconds=25), 2.5, 2.0)
    ck(gate.remaining("s1", g0 + timedelta(seconds=30)) == 20,
       "出现 > 阈值观测后连续计时清零")
    gate.observe("s2", g0, 1.0, 2.0)
    gate.observe("s2", g0 + timedelta(seconds=5), None, 2.0)
    ck(gate.remaining("s2", g0 + timedelta(seconds=10)) == 20,
       "数据缺失中断连续计时")
    gate.observe("s3", g0, 1.0, 2.0)
    gate.observe("s3", g0 + timedelta(seconds=60), 1.0, 2.0)
    ck(gate.remaining("s3", g0 + timedelta(seconds=65)) == 20,
       "观测间隔超过 max_gap 视为中断（非盘中/午休不误放行）")
    gate.reset("s1")
    ck(gate.remaining("s1", g0 + timedelta(seconds=100)) == 20, "reset 清空前置历史")

    pe = AlertEngine(2.0, 1.0, day_precheck_sec=20, max_gap_sec=15)
    p0 = datetime(2026, 11, 9, 10, 0, tzinfo=BJ)

    def pstep(sec, day, min1=None, sym="s1"):
        return pe.evaluate(sym, "测试", 100 + day, day, min1,
                           p0 + timedelta(seconds=sec))

    ck(pstep(0, 1.0) is None and pstep(5, 1.0) is None
       and pstep(10, 1.5) is None, "阈值下方观测不触发")
    ck(pstep(15, 2.5) is None, "未满前置窗口时上穿被抑制")
    ck(not pe.locked.get(("s1", "day"), False), "抑制不消耗触发机会（不上锁）")
    ck(pstep(20, 2.6) is None and pstep(35, 2.7) is None,
       "持续站在阈值上方不会补发提醒")
    for s in (40, 45, 50, 55, 60):
        pstep(s, 1.0)
    fired = pstep(65, 2.5)
    ck(fired is not None and "当日涨幅" in fired.conditions,
       "回落后重新满足前置窗口，上穿才触发")
    ck(pe.locked[("s1", "day")] is True, "真正触发后才上锁")
    ck(pstep(70, 2.6) is None, "上锁后仍不重复提醒")

    only_min1 = AlertEngine(2.0, 1.0, day_precheck_sec=300, max_gap_sec=15)
    m = only_min1.evaluate("s2", "测试2", 102.5, 2.5, 1.5, p0)
    ck(m is not None and "近1分钟涨幅" in m.conditions
       and "当日涨幅" not in m.conditions,
       "day 被抑制时 min1 仍单独触发，合并文本只含 min1")
    ck(not only_min1.locked.get(("s2", "day"), False), "被抑制的 day 不上锁")
    ck(only_min1.locked.get(("s2", "min1"), False) is True, "min1 正常上锁")

    eq = AlertEngine(2.0, 1.0, day_precheck_sec=10, max_gap_sec=15)
    eq.evaluate("s3", "测试3", 102.0, 2.0, None, p0)
    ck(eq.evaluate("s3", "测试3", 102.5, 2.5, None,
                   p0 + timedelta(seconds=12)) is not None,
       "== 阈值视为下方，满窗口后上穿触发")
    off = AlertEngine(2.0, 1.0, day_precheck_sec=0)
    ck(off.evaluate("s4", "测试4", 102.5, 2.5, None, p0) is not None,
       "precheck_sec=0 回滚为上穿即提醒")

    # 慢取数（主源超时 + 备用源也慢）时放宽 max_gap，不该误清零连续下方计时
    wide = DayChangeGate(precheck_sec=20, max_gap_sec=30)
    wide.observe("s1", g0, 1.0, 2.0)
    wide.observe("s1", g0 + timedelta(seconds=25), 1.0, 2.0)
    ck(wide.remaining("s1", g0 + timedelta(seconds=30)) == 0,
       "放宽 max_gap 后慢取数不会误清零前置窗口")

    # 7. 配置校验
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    validate_config(cfg)
    ck(cfg["stocks"][0]["symbol"] == normalize_symbol(cfg["stocks"][0]["code"]),
       "规范代码写入 symbol")
    bad = copy.deepcopy(DEFAULT_CONFIG)
    bad["stocks"] = bad["stocks"][:2]
    try:
        validate_config(bad)
        ck(False, "股票数量 <3 应报错")
    except ValueError:
        ck(True, "股票数量 <3 报错")
    bad2 = copy.deepcopy(DEFAULT_CONFIG)
    bad2["day_change_threshold_pct"] = 0
    try:
        validate_config(bad2)
        ck(False, "阈值 <=0 应报错")
    except ValueError:
        ck(True, "阈值 <=0 报错")
    bad_precheck = copy.deepcopy(DEFAULT_CONFIG)
    bad_precheck["day_change_precheck_sec"] = 901
    try:
        validate_config(bad_precheck)
        ck(False, "day_change_precheck_sec 越界应报错")
    except ValueError:
        ck(True, "day_change_precheck_sec 越界报错")
    ck(DEFAULT_CONFIG["day_change_precheck_sec"] == 300,
       "当日涨幅前置窗口默认 300 秒（前 5 分钟）")
    ck(normalize_symbol("600519") == "sh600519", "纯 6 位代码前缀推断")

    # 8. 格式化
    ck(fmt_price(1448.27) == "¥1,448.27", "价格格式化")
    ck(fmt_pct(2.35) == "+2.35%" and fmt_pct(-0.33) == "-0.33%", "涨跌幅格式化")

    # 9. MACD 计算：same_day 初值、EMA 方向、柱口径、两档预热差异
    mcfg = copy.deepcopy(DEFAULT_CONFIG)
    mcfg["macd"]["warmup"] = "same_day"        # 第 9 节固定用当日序列口径做数值对照
    mcfg["macd"]["warmup_bars"] = 1            # 单测用，绕过 35 根门槛
    day = "2026-11-09"

    def mk_bars(closes, trade_day=day):
        return tuple(MinuteBar(trade_day, f"{i:04d}", c)
                     for i, c in enumerate(closes))

    flat = compute_macd(mk_bars([100.0] * 10), mcfg)
    ck(len(flat) == 10 and flat[0].dif == 0.0 and flat[0].dea == 0.0,
       "same_day 首根 DIF/DEA 初值为 0")
    ck(all(p.bar == 0.0 for p in flat), "常数序列无交叉、柱恒为 0")
    up = compute_macd(mk_bars([100.0] + [100.0 + i for i in range(1, 20)]), mcfg)
    down = compute_macd(mk_bars([100.0] + [100.0 - i for i in range(1, 20)]), mcfg)
    ck(up[-1].dif > 0 and down[-1].dif < 0, "EMA12/26 方向正确")
    ck(abs(up[-1].bar - 2.0 * (up[-1].dif - up[-1].dea)) < 1e-9,
       "柱 = 2×(DIF−DEA)（A 股口径）")
    prev_bars = mk_bars([100.0] * 5, "2026-11-06") + mk_bars([100.0] * 5)
    p_same = compute_macd(prev_bars, mcfg)
    mcfg_prev = copy.deepcopy(mcfg)
    mcfg_prev["macd"]["warmup"] = "prev_day"
    p_prev = compute_macd(prev_bars, mcfg_prev)
    ck(len(p_same) == 5 and len(p_prev) == 10, "两档预热选取的序列范围不同")
    ck(p_same[0].dif == 0.0, "same_day 不继承前日 EMA 状态")

    # 10. 交叉判定与 0 轴顺势过滤（直接构造指标点，精确可控）
    def pt(dif, dea, index):
        return MacdPoint(day, f"{index:04d}", 100.0, dif, dea,
                         2.0 * (dif - dea))

    pts = [pt(0, 0, 0), pt(0.20, 0.05, 1)]
    cr = detect_cross(pts, mcfg, "s1")
    ck(len(cr) == 1 and cr[0].direction == "golden", "DIF 上穿 DEA 判金叉")
    cr_d = detect_cross([pt(0, 0, 0), pt(-0.20, -0.05, 1)], mcfg, "s1")
    ck(len(cr_d) == 1 and cr_d[0].direction == "death", "DIF 下穿 DEA 判死叉")
    reverse = [pt(0, 0, 0), pt(-0.20, -0.10, 1), pt(-0.05, -0.10, 2)]
    ck(not any(c.direction == "golden" for c in detect_cross(reverse, mcfg, "s1")),
       "0 轴过滤拒绝 DIF<0 的金叉")
    mcfg_nofilter = copy.deepcopy(mcfg)
    mcfg_nofilter["macd"]["zero_axis_filter"] = False
    ck(any(c.direction == "golden" for c in detect_cross(reverse, mcfg_nofilter, "s1")),
       "关闭 0 轴过滤后该金叉可见")
    ck(not detect_cross([pt(0, 0, 0)], mcfg, "s1"), "单根不构成交叉")
    cross_day = [MacdPoint("2026-11-06", "0001", 100, 0.2, 0.05, 0.3),
                 MacdPoint(day, "0002", 100, -0.2, -0.05, -0.3)]
    ck(not detect_cross(cross_day, mcfg, "s1"),
       "跨日两根不比较（午休/隔夜断点不产生伪交叉）")
    mcfg_glue = copy.deepcopy(mcfg)
    mcfg_glue["macd"]["min_dif_ratio"] = 0.01    # 现价 100 → |DIF| 需 ≥ 1
    glue = [pt(0, 0, 0), pt(0.02, 0.01, 1)]
    ck(not detect_cross(glue, mcfg_glue, "s1"),
       "0 轴黏合交叉被相对阈值过滤")
    mcfg_glue["macd"]["min_dif_ratio"] = 0.0
    ck(len(detect_cross(glue, mcfg_glue, "s1")) == 1,
       "关闭黏合过滤后该交叉可见")

    # 11. 离散信号状态机：柱确认、去重、冷却、上限、跨日、换源、门控
    def points_from(values):
        return [pt(d, e, i) for i, (d, e) in enumerate(values)]

    def run_points(engine, cfg, values, warmup_override=None):
        c = copy.deepcopy(cfg)
        if warmup_override is not None:
            c["macd"]["warmup_bars"] = warmup_override
        seq = points_from(values)
        t0 = datetime(2026, 11, 9, 9, 31, tzinfo=BJ)
        got = []
        for i in range(1, len(seq) + 1):
            part = seq[:i]
            frame = MacdSymbolFrame("s1", True, "sina_kline", day, (),
                                    part[-1].minute)
            got += engine.process("s1", frame, part,
                                  detect_cross(part, c, "s1", day),
                                  t0 + timedelta(minutes=i),
                                  {"name": "测试", "price": 100.0,
                                   "day": 0.0, "min1": 0.0}, True, False)
        return got

    confirmed = [(0, 0)] * 3 + [(0.10, 0.05), (0.20, 0.08), (0.30, 0.10)] + [(0.30, 0.10)] * 2
    e1 = MacdSignalEngine(mcfg)
    got1 = run_points(e1, mcfg, confirmed, 1)
    ck(len(got1) == 1 and got1[0].kind == "macd" and got1[0].direction == "golden",
       "柱连续 2 根同向放大确认后发出一条金叉")
    rejected = [(0, 0)] * 3 + [(0.10, 0.05), (0.12, 0.10), (0.30, 0.10)] + [(0.30, 0.10)] * 2
    ck(not run_points(MacdSignalEngine(mcfg), mcfg, rejected, 1),
       "柱缩小即丢弃候选，不发提醒")
    mcfg_zero = copy.deepcopy(mcfg)
    mcfg_zero["macd"]["bar_confirm_bars"] = 0
    ck(len(run_points(MacdSignalEngine(mcfg_zero), mcfg_zero,
                      [(0, 0)] * 2 + [(0.10, 0.05)], 1)) == 1,
       "bar_confirm_bars=0 时交叉即报")

    # on_new_day 全局清理：只认交易日前进，防止停牌/滞后源回吐前一日反复清空
    e_day = MacdSignalEngine(mcfg)
    ck(e_day.on_new_day("2026-11-09") is True, "首个交易日建立全局状态")
    e_day._seen.add(("s1", "2026-11-09", "0001", "golden"))
    ck(e_day.on_new_day("2026-11-09") is False and e_day._seen,
       "同一交易日不重复清空去重集合")
    ck(e_day.on_new_day("2026-11-06") is False and bool(e_day._seen),
       "回吐前一交易日不得倒退清空去重集合")
    ck(e_day.on_new_day("2026-11-10") is True and not e_day._seen,
       "交易日前进才清空全局状态")

    e2 = MacdSignalEngine(mcfg)
    t0 = datetime(2026, 11, 9, 9, 31, tzinfo=BJ)     # 上午（限流未生效）
    t_pm = datetime(2026, 11, 9, 13, 0, tzinfo=BJ)   # 下午（限流生效）

    def cross(minute, direction="golden"):
        return MacdCross("s1", day, minute, direction, 0.2, 0.05, 0.3,
                         100.0, "same_day")

    ck(e2._allow_emit("s1", cross("0001"), t_pm), "下午首次信号放行")
    ck(not e2._allow_emit("s1", cross("0001"), t_pm + timedelta(minutes=1)),
       "同一 (股,日,分钟,方向) 不重复提醒")
    ck(not e2._allow_emit("s1", cross("0002"), t_pm + timedelta(minutes=5)),
       "下午冷却 15 分钟内抑制")
    ck(e2._allow_emit("s1", cross("0003"), t_pm + timedelta(minutes=16)),
       "下午冷却结束后恢复")
    ck(e2._allow_emit("s1", cross("0004"), t_pm + timedelta(minutes=32)),
       "下午第三次信号放行")
    ck(not e2._allow_emit("s1", cross("0005"), t_pm + timedelta(minutes=48)),
       "下午达到每日上限 3 次后抑制")
    e2.on_new_day("2026-11-10")
    next_day_cross = MacdCross("s1", "2026-11-10", "0006", "golden",
                               0.2, 0.05, 0.3, 100.0, "same_day")
    ck(e2._allow_emit("s1", next_day_cross, t_pm + timedelta(days=1)),
       "跨日清计数与去重集合")

    # 11b. 冷却 / 每日上限「上午无效、下午生效」
    ck(not macd_throttle_active(t0, DEFAULT_CONFIG)
       and not macd_throttle_active(datetime(2026, 11, 9, 11, 29, tzinfo=BJ),
                                   DEFAULT_CONFIG)
       and macd_throttle_active(t_pm, DEFAULT_CONFIG)
       and macd_throttle_active(datetime(2026, 11, 9, 14, 30, tzinfo=BJ),
                                DEFAULT_CONFIG),
       "限流默认上午无效、13:00 起生效")
    e7 = MacdSignalEngine(copy.deepcopy(DEFAULT_CONFIG))
    for i in range(5):
        ck(e7._allow_emit("s1", cross(f"100{i}"), t0 + timedelta(minutes=i * 3)),
           f"上午第 {i + 1} 条信号不受冷却/每日上限限制")
    ck(e7.rt["s1"].badge == BADGE_GOLDEN and not e7._daily
       and not e7._last_emit,
       "上午不累计下午额度、不启动冷却计时")
    ck(e7._allow_emit("s1", cross("1301"), t_pm),
       "下午第一条不受上午信号影响，额度重新起算")
    ck(not e7._allow_emit("s1", cross("1302"), t_pm + timedelta(minutes=1))
       and e7.rt["s1"].badge == "冷却中",
       "下午冷却生效")
    e8_cfg = copy.deepcopy(DEFAULT_CONFIG)
    e8_cfg["macd"]["count_morning_in_daily_max"] = True
    e8 = MacdSignalEngine(e8_cfg)
    for i in range(4):
        e8._allow_emit("s1", cross(f"110{i}"), t0 + timedelta(minutes=i))
    ck(not e8._allow_emit("s1", cross("1305"), t_pm)
       and e8.rt["s1"].badge == "今日已满",
       "count_morning_in_daily_max=true 时上午信号占用下午额度")
    all_day_cfg = copy.deepcopy(DEFAULT_CONFIG)
    all_day_cfg["macd"]["throttle_start"] = "09:30"
    e9 = MacdSignalEngine(all_day_cfg)
    ck(macd_throttle_active(t0, all_day_cfg)
       and e9._allow_emit("s1", cross("0932"), t0)
       and not e9._allow_emit("s1", cross("0933"), t0 + timedelta(minutes=1)),
       "throttle_start=09:30 可恢复全天生效的旧行为")
    v_am = macd_cross_to_view(cross("1000"), {}, DEFAULT_CONFIG, t0)
    v_pm = macd_cross_to_view(cross("1301"), {}, DEFAULT_CONFIG, t_pm)
    ck("上午不限冷却" in v_am.confirm_note
       and "15 分钟内同股不重复提示" in v_pm.confirm_note
       and v_am.confirm_note.split(" · ")[0] == v_pm.confirm_note.split(" · ")[0],
       "提醒文案随上午/下午限流模式切换，弹窗首段不变")
    seen_before = len(e2._seen)
    e2.reset_compute("s1", "eastmoney_trends2")
    ck(len(e2._seen) == seen_before and len(e2._daily) > 0,
       "换源只清计算态，保留去重/计数")
    e3 = MacdSignalEngine(mcfg)
    frame_ok = MacdSymbolFrame("s1", True, "sina_kline", day, (), "0001")
    pts_ok = [pt(0, 0, 0)] * 40
    ck(not e3.process("s1", frame_ok, pts_ok, [], t0, None, False, False)
       and e3.rt["s1"].badge == "静默", "非交易时段冻结不发信号")
    ck(not e3.process("s1", frame_ok, pts_ok, [], t0, None, True, True),
       "静态行情不发信号")
    fail_frame = MacdSymbolFrame("s1", False, "sina_kline", day, (), None,
                                 False, "模拟失败")
    ck(not e3.process("s1", fail_frame, [], [], t0, None, True, False)
       and e3.rt["s1"].badge == "分钟线失败", "分钟线失败只更新角标")
    e4 = MacdSignalEngine(copy.deepcopy(DEFAULT_CONFIG))   # warmup_bars=35
    short = [pt(0, 0, i) for i in range(3)]
    ck(not e4.process("s1", frame_ok, short, [], t0, None, True, False)
       and e4.rt["s1"].badge.startswith("预热"), "预热不足显示预热进度")
    e6_cfg = copy.deepcopy(DEFAULT_CONFIG)
    e6_cfg["macd"]["warmup"] = "same_day"
    e6 = MacdSignalEngine(e6_cfg)
    p34 = [pt(0, 0, i) for i in range(34)]
    p35 = [pt(0, 0, i) for i in range(35)]
    e6.process("s1", MacdSymbolFrame("s1", True, "sina_kline", day, (), "0034"),
               p34, [], t0, None, True, False)
    ck(e6.rt["s1"].badge.startswith("预热"), "预热期显示进度")
    e6.process("s1", MacdSymbolFrame("s1", True, "sina_kline", day, (), "0035"),
               p35, [], t0, None, True, False)
    ck(e6.rt["s1"].badge == "", "预热达标后清掉陈旧进度角标")

    # 默认 prev_day：跨日序列已充分递推，当日仅 1 根也已预热达标（开盘即有效）
    e5 = MacdSignalEngine(copy.deepcopy(DEFAULT_CONFIG))
    hist = [MacdPoint("2026-11-06", f"{i:04d}", 100.0, 0.0, 0.0, 0.0)
            for i in range(40)]
    day1 = [MacdPoint(day, "0931", 100.0, 0.0, 0.0, 0.0)]
    frame_early = MacdSymbolFrame("s1", True, "sina_kline", day, (), "0931")
    ck(not e5.process("s1", frame_early, hist + day1, [], t0, None, True, False)
       and e5.rt["s1"].warmup_ok,
       "prev_day 当日仅 1 根也已预热达标（开盘即有效）")

    # 非盘中零联网开关
    off_cfg = copy.deepcopy(DEFAULT_CONFIG)
    off_cfg["fetch_only_in_session"] = False
    after = datetime(2026, 11, 9, 16, 0, tzinfo=BJ)
    intraday = datetime(2026, 11, 9, 10, 0, tzinfo=BJ)
    ck(not macd_http_allowed(after, DEFAULT_CONFIG)
       and macd_http_allowed(intraday, DEFAULT_CONFIG)
       and macd_http_allowed(after, off_cfg),
       "fetch_only_in_session 控制盘后是否请求分钟线")
    ck(session_of(intraday) == "盘中"
       and session_of(datetime(2026, 11, 9, 9, 20, tzinfo=BJ)) != "盘中"
       and session_of(datetime(2026, 11, 9, 12, 0, tzinfo=BJ)) != "盘中",
       "价格线程暂停判定覆盖集合竞价与午休")
    ck(macd_in_session_windows(datetime(2026, 11, 9, 9, 32, tzinfo=BJ),
                               DEFAULT_CONFIG),
       "时段窗口覆盖开盘后早期分钟")

    # 12. 分钟线解析、脏数据防御、源切换、secid 映射
    ck(to_eastmoney_secid("sh600519") == "1.600519", "secid sh→1.")
    ck(to_eastmoney_secid("sz000001") == "0.000001", "secid sz→0.")
    ck(to_eastmoney_secid("bj430047") == "0.430047", "secid bj→0.（实测结论）")
    ck("datalen=1200" in build_sina_kline_url("sh600519", 1200), "新浪 K 线 URL")
    ck("ndays=5" in build_eastmoney_trends2_url("sh600519", DEFAULT_CONFIG),
       "东财 trends2 URL")
    try:
        parse_sina_kline("null", "sh600519")
        ck(False, "新浪返回 null 应报错")
    except ValueError:
        ck(True, "新浪返回 null 判空")
    sina_json = json.dumps([
        {"day": "2026-11-09 09:31:00", "close": "1441.990"},
        {"day": "2026-11-09 09:32:00", "close": "bad"},
        {"day": "2026-11-09 15:00:00", "close": "1444.990"},
    ])
    sina_bars = parse_sina_kline(sina_json, "sh600519")
    ck(len(sina_bars) == 2 and sina_bars[0].minute == "0931"
       and sina_bars[1].close == 1444.99, "新浪解析：字符串 close / 单根异常隔离")
    em_json = json.dumps({"data": {"trends": [
        "2026-11-09 09:30,1440.06,1440.06,1440.06,1440.06,186,26819677,1440.060",
        "2026-11-09 15:00,1444.01,1444.99,1444.99,1444.01,854,123402146,1442.835",
    ]}})
    em_bars = parse_eastmoney_trends2(em_json, "sh600519")
    ck(len(em_bars) == 2 and em_bars[1].close == 1444.99
       and em_bars[0].minute == "0930", "东财解析：[0] 时间 / [2] 收盘价")
    try:
        parse_eastmoney_trends2('{"data":null}', "sh600519")
        ck(False, "东财 data:null 应报错")
    except ValueError:
        ck(True, "东财 data:null 判空")
    dirty = [MinuteBar(day, "1506", 100.0), MinuteBar(day, "0931", 100.0),
             MinuteBar(day, "0931", 100.5), MinuteBar("2026-11-06", "1500", 99.0)]
    norm, nday, nlast = normalize_bars(dirty, "1500")
    ck(nday == day and nlast == "0931" and len(norm) == 2
       and norm[1].close == 100.5, "截断盘后脏点 / 去重留末根 / 分日排序")
    switch_cfg = copy.deepcopy(DEFAULT_CONFIG)
    switch_cfg["macd"]["retry"] = 0

    def fake_fetcher(url):
        if "sina" in url:
            raise OSError("模拟主源不可达")
        return em_json

    sf = fetch_minute_bars("sh600519", switch_cfg, fake_fetcher, now_hhmm="1500")
    ck(sf.ok and sf.source == "eastmoney_trends2" and sf.used_fallback,
       "主源失败整段切换备用源")
    sf_fail = fetch_minute_bars("sh600519", switch_cfg,
                                lambda _url: (_ for _ in ()).throw(OSError("全断")),
                                now_hhmm="1500")
    ck(not sf_fail.ok and bool(sf_fail.error),
       "两源都失败返回 ok=False 与具体原因")

    # 换源游标：同分钟换源必须投递，失败帧只投递不写游标，时间倒退不回退
    cur = {}
    f_ok = MacdSymbolFrame("s1", True, "sina_kline", "2026-11-09",
                           (), "1407", False)
    deliver, key = macd_cursor_advance(cur, "s1", f_ok)
    ck(deliver and key is not None, "首帧投递并写入游标")
    cur["s1"] = key
    ck(macd_cursor_advance(cur, "s1", f_ok) == (False, None),
       "同一分钟同源不重复投递")
    f_switch = MacdSymbolFrame("s1", True, "eastmoney_trends2", "2026-11-09",
                               (), "1407", True)
    ck(macd_cursor_advance(cur, "s1", f_switch)[0] is True,
       "换源帧即使同分钟也必须投递（否则 reset_compute 永不触发）")
    f_back = MacdSymbolFrame("s1", True, "sina_kline", "2026-11-09",
                             (), "1400", False)
    ck(macd_cursor_advance(cur, "s1", f_back)[0] is False,
       "时间倒退不投递、不回退游标")
    snapshot_cursor = dict(cur)
    f_fail = MacdSymbolFrame("s1", False, "sina_kline", None, (), None,
                             False, "全断")
    ck(macd_cursor_advance(cur, "s1", f_fail) == (True, None)
       and cur == snapshot_cursor, "失败帧投递但不覆盖有效游标")

    # 开盘当日只有 1 根 K 线不能被判成「停牌全平」；多根同价仍要判停牌
    one_bar_cfg = copy.deepcopy(DEFAULT_CONFIG)
    one_bar_cfg["macd"]["retry"] = 0
    one_bar_json = json.dumps([
        {"day": "2026-11-06 15:00:00", "close": "99.0"},
        {"day": "2026-11-09 09:31:00", "close": "100.0"},
    ])
    sf_one = fetch_minute_bars("sh600519", one_bar_cfg,
                               lambda _u: one_bar_json, now_hhmm="0931")
    ck(sf_one.ok and sf_one.last_minute == "0931",
       "开盘当日仅 1 根 K 线不再被误判为停牌全平")
    flat_json = json.dumps([
        {"day": f"2026-11-09 09:{m}:00", "close": "8.17"}
        for m in (31, 32, 33)
    ])
    sf_flat = fetch_minute_bars("sh600519", one_bar_cfg,
                                lambda _u: flat_json, now_hhmm="0933")
    ck(not sf_flat.ok and "全平" in sf_flat.error,
       "当日多根同价仍判定为停牌全平")

    # 双源全失败时不能谎报主源成功
    original_fetch = globals()["_fetch_text"]
    globals()["_fetch_text"] = lambda *a, **k: (
        _ for _ in ()).throw(OSError("模拟双源全断"))
    try:
        fail_cfg = copy.deepcopy(DEFAULT_CONFIG)
        fail_cfg["request_timeout_sec"] = 1
        bf = fetch_quotes(["sh600519"], fail_cfg, 1)
    finally:
        globals()["_fetch_text"] = original_fetch
    ck(bf.source == "sina" and bf.used_fallback and "__all__" in bf.errors,
       "双源全失败：source 取最后尝试的源并标记已切备用")

    # 13. 配置浅合并 / 校验 / 合规文案回归 / 统一提醒视图
    import tempfile
    partial = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                          encoding="utf-8")
    json.dump({"macd": {"enabled": False}}, partial)
    partial.close()
    cfg_partial, _ = load_config(partial.name)
    os.remove(partial.name)
    ck(cfg_partial["macd"]["enabled"] is False
       and cfg_partial["macd"]["warmup"] == "prev_day"
       and cfg_partial["macd"]["daily_max"] == 3
       and cfg_partial["fetch_only_in_session"] is True,
       "macd 子对象浅合并 + 顶层开关保留默认值")
    bad_macd = copy.deepcopy(DEFAULT_CONFIG)
    bad_macd["macd"]["warmup"] = "bad"
    try:
        validate_config(bad_macd)
        ck(False, "非法 warmup 应报错")
    except ValueError:
        ck(True, "非法 warmup 枚举报错")
    bad_macd2 = copy.deepcopy(DEFAULT_CONFIG)
    bad_macd2["macd"]["ema_slow"] = 10
    try:
        validate_config(bad_macd2)
        ck(False, "ema_slow<=ema_fast 应报错")
    except ValueError:
        ck(True, "ema_slow<=ema_fast 报错")
    bad_macd3 = copy.deepcopy(DEFAULT_CONFIG)
    bad_macd3["macd"]["min_dif_ratio"] = 1
    try:
        validate_config(bad_macd3)
        ck(False, "min_dif_ratio 越界应报错")
    except ValueError:
        ck(True, "min_dif_ratio 越界报错")
    bad_macd4 = copy.deepcopy(DEFAULT_CONFIG)
    bad_macd4["macd"]["throttle_start"] = "25:99"
    try:
        validate_config(bad_macd4)
        ck(False, "throttle_start 非法应报错")
    except ValueError:
        ck(True, "throttle_start 非法报错")
    bad_macd5 = copy.deepcopy(DEFAULT_CONFIG)
    bad_macd5["macd"]["count_morning_in_daily_max"] = "false"
    try:
        validate_config(bad_macd5)
        ck(False, "count_morning_in_daily_max 非布尔应报错")
    except ValueError:
        ck(True, "count_morning_in_daily_max 非布尔报错")
    ck(cfg_partial["macd"]["throttle_start"] == "13:00"
       and cfg_partial["macd"]["count_morning_in_daily_max"] is False,
       "新限流键未写配置时取默认值")
    all_text = " ".join([MACD_DISCLAIMER, MACD_METHOD_NOTE, MACD_TPLUS1_NOTE,
                         BADGE_GOLDEN, BADGE_DEATH])
    ck(not any(w in all_text for w in MACD_FORBIDDEN_WORDS),
       "固定文案不含任何买卖动作词")
    ck("不构成投资建议" in MACD_DISCLAIMER, "免责声明保留")
    a_view = alert_to_view(Alert("s1", "测试", 100.0, 2.5, 1.2,
                                 "当日涨幅 > 2.00%", t0))
    ck(a_view.kind == "threshold" and a_view.conditions == "当日涨幅 > 2.00%"
       and a_view.day_pct == 2.5, "涨幅告警转统一视图字段不变")
    m_view = macd_cross_to_view(cross("0001"), {"name": "测试", "price": 100.0,
                                               "day": 1.0, "min1": 0.5},
                                DEFAULT_CONFIG, t0, "sina_kline")
    ck(m_view.kind == "macd" and "金叉" in m_view.conditions
       and not any(w in m_view.conditions for w in MACD_FORBIDDEN_WORDS),
       "MACD 视图只报指标状态")
    fr_mock = mock_minute_frame("s1", DEFAULT_CONFIG, 3)
    ck(fr_mock.per_symbol["s1"].ok
       and len(fr_mock.per_symbol["s1"].bars) == 3,
       "mock 分钟帧逐根推进")
    ck(not mock_minute_frame("s1", {**DEFAULT_CONFIG, "macd": {
        **DEFAULT_CONFIG["macd"], "mock_scenario": "null_and_dirty"}}, 3)
       .per_symbol["s1"].ok, "停牌/脏数据 mock 帧返回 ok=False")

    print(f"SELFTEST OK: {checks} checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    print("用法: python core.py --selftest  （本模块为 app.py 的逻辑层，不单独运行）")
    sys.exit(0)
