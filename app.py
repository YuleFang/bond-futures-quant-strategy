# -*- coding: utf-8 -*-
"""股票行情监控小窗口 —— 界面与调度层（磨砂玻璃 / Skeuomorphic Glass 风格）。

启动（务必用带 tkinter 的 Python 3.11，详见 README.md）：
    python app.py
"""
from __future__ import annotations

import logging
import os
import queue
import re
import sys
import threading
import time
from collections import deque

try:
    import tkinter as tk
except ImportError:
    print("当前 Python 缺少 tkinter。请在 PowerShell 中使用 Python 3.11 启动（详见 README.md）。",
          file=sys.stderr)
    sys.exit(2)

import core

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "config.json")

# ---------------------------------------------------------------------------
# 玻璃拟态配色（暖米色玻璃 + 近黑文字 + 橙色柔光）
# A 股配色惯例保留：涨红、跌绿（这里用与暖调协调的暖红 / 墨绿）
# ---------------------------------------------------------------------------
KEY = "#e7dece"          # 色键透明色；万一平台不支持透明色，四角仍是柔和米色
GLASS = "#f2ece3"        # 玻璃主面板
GLASS_HI = "#fbf7f1"     # 胶囊 / 高光
GLASS_DIM = "#e9e2d7"    # 失败置灰时的胶囊
GLASS_EDGE = "#fffdf9"   # 玻璃边缘高光
INK = "#2c2a26"          # 主文字
INK_SOFT = "#746d62"     # 次级文字
INK_FAINT = "#a79e91"    # 辅助文字
ACCENT = "#e08a3c"       # 橙色强调
UP = "#b8412f"           # 涨（暖红）
DOWN = "#2f7a52"         # 跌（墨绿）
WARN = "#c86f2c"         # 警告 / 延迟
DIM = "#a9a196"          # 失败、延迟整行置灰

FONT_CN = "Microsoft YaHei UI"
FONT_NUM = "Segoe UI"

SOURCE_LABEL = {"tencent": "腾讯", "sina": "新浪", "mock": "模拟"}
HINT_TEXT = "当日=(现价−昨收)/昨收   ·   近1分=(现价−1分前快照)/该快照"

ALERT_W, ALERT_H = 340, 380         # 提醒窗尺寸（MACD 状态需多 4 行）
ALERT_MARGIN_X, ALERT_MARGIN_Y = 24, 56   # 右下角边距（Y 避开任务栏）
ALERT_GAP = 12                      # 多条提醒向上堆叠的间距
MACD_BADGE_X = 198                  # 行内 MACD 角标中心 x（r3 中部空白，不新增行）
PAD = 18
TOP_H = 128                         # 顶部标签 / 标题 / 大号时间区
BOTTOM_H = 56                       # 底部状态胶囊
HINT_H = 20                         # 口径说明行


def _wrap_disclaimer(value: str, limit: float = 26.0) -> list[str]:
    """把免责全文按「 · 」分段折行，保证完整文案不被裁切、不出现省略号。

    中文按 1 个视觉单位、ASCII 按 0.55 估算；只做分段级换行，
    不会把一个词组从中间劈开。core.MACD_DISCLAIMER 保持为单一常量不变。
    """
    def visual_len(s: str) -> float:
        return sum(1.0 if ord(ch) > 127 else 0.55 for ch in s)

    lines: list[str] = []
    cur = ""
    for part in value.split(" · "):
        candidate = part if not cur else cur + " · " + part
        if cur and visual_len(candidate) > limit:
            lines.append(cur)
            cur = part
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines


def setup_logging(cfg: dict) -> None:
    level = getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO)
    handlers = [logging.StreamHandler()]
    if cfg.get("log_to_file"):
        handlers.append(logging.FileHandler(os.path.join(BASE, "monitor.log"),
                                           encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(message)s")


def _round_rect(canvas, x1, y1, x2, y2, radius, **kwargs):
    """用平滑多边形画圆角矩形（tkinter 没有原生圆角控件）。"""
    r = min(radius, (x2 - x1) / 2, (y2 - y1) / 2)
    points = (
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    )
    return canvas.create_polygon(points, smooth=True, **kwargs)


def _soft_glow(canvas, cx, cy, radius):
    """用多层平滑色阶同心圆模拟橙色柔光球（tkinter 无模糊，靠色阶近似）。"""
    layers = (
        (1.34, "#f8ead7"),
        (1.20, "#f7e0c2"),
        (1.06, "#f5d4a9"),
        (0.93, "#f2c691"),
        (0.81, "#efb679"),
        (0.69, "#eba562"),
        (0.57, "#e69449"),
    )
    for scale, color in layers:
        r = radius * scale
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=color, outline="")


class MonitorApp:
    def __init__(self, root: tk.Tk, cfg: dict, config_note: str = ""):
        self.root = root
        self.cfg = cfg
        self.config_note = config_note
        self.symbols = [item["symbol"] for item in cfg["stocks"]]
        self.names = {item["symbol"]: (item.get("name") or item["symbol"])
                      for item in cfg["stocks"]}

        # 所有状态机只由主线程访问；后台线程仅 put 不可变 Batch，无需 Lock
        self.snapshots = core.SnapshotStore(cfg["snapshot_window_sec"],
                                            cfg["refresh_interval_sec"])
        # 连续观测允许的间隔必须能容纳「主源超时 + 备用源也慢」的最坏一轮，
        # 否则一次慢取数就会把「前 5 分钟」窗口清零、让当日涨幅告警静默 5 分钟。
        self.engine = core.AlertEngine(
            cfg["day_change_threshold_pct"],
            cfg["min1_change_threshold_pct"],
            cfg["day_change_precheck_sec"],
            max_gap_sec=max(15.0,
                            3 * cfg["refresh_interval_sec"],
                            2 * cfg["request_timeout_sec"]
                            + cfg["refresh_interval_sec"] + 2))
        self.states = {
            sym: {"name": self.names[sym], "price": None, "day": None, "min1": None,
                  "min1_text": "—", "status": "等待数据", "dim": True, "warn": False}
            for sym in self.symbols
        }
        # MACD 是独立的离散事件链路，不复用 AlertEngine 的连续阈值锁
        self.macd_engine = core.MacdSignalEngine(cfg)
        self.macd_states = {sym: ("", INK_FAINT) for sym in self.symbols}
        self.last_success = None
        self.last_quote_ts = None
        self.same_ts_count = 0
        self.connection = "等待数据"
        self.session = "—"
        self.source_in_use = cfg["data_source"]

        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.alert_queue: deque = deque()
        self.open_alerts = 0
        self.alert_slots: set = set()      # 右下角堆叠位置槽，避免重叠
        self.after_id = None
        self.closed = False
        self._drag_origin = None
        self.row_items = {}

        self._build_ui()
        root.report_callback_exception = self._on_tk_error

        self.thread = threading.Thread(target=self._fetch_loop, daemon=True,
                                       name="quote-fetcher")
        self.thread.start()
        # 第二路分钟线取数：独立 daemon，慢 HTTP 不会阻塞 5 秒快照刷新
        self.macd_thread = threading.Thread(target=self._macd_fetch_loop, daemon=True,
                                            name="macd-fetcher")
        self.macd_thread.start()
        self.after_id = root.after(100, self._poll_queue)

    # ------------------------------------------------------------------ UI
    @staticmethod
    def _apply_glass(win, alpha: float = 0.975) -> bool:
        """无边框 + 色键透明圆角 + 整体微透明；失败则降级为实色普通窗口。"""
        win.configure(bg=KEY)
        try:
            win.overrideredirect(True)
            win.attributes("-transparentcolor", KEY)
            win.attributes("-alpha", alpha)
            return True
        except tk.TclError:
            win.overrideredirect(False)
            win.configure(bg=GLASS)
            return False

    def _build_ui(self) -> None:
        root = self.root
        match = re.fullmatch(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
                             str(self.cfg["window_geometry"]))
        width, height, pos_x, pos_y = (tuple(map(int, match.groups()))
                                       if match else (380, 560, 120, 120))
        self.width, self.height = width, height
        root.title("股票行情监控")
        root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        self.glass = self._apply_glass(root)
        # overrideredirect 会重设窗口样式，置顶必须在其后重新设置
        root.attributes("-topmost", bool(self.cfg["always_on_top"]))
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Escape>", lambda _event: self.on_close())

        canvas = tk.Canvas(root, width=width, height=height,
                           bg=(KEY if self.glass else GLASS),
                           highlightthickness=0, bd=0)
        canvas.pack()
        self.canvas = canvas

        # 主玻璃面板 + 顶部柔光
        _round_rect(canvas, 2, 2, width - 2, height - 2, 26,
                    fill=GLASS, outline=GLASS_EDGE, width=1.5)
        _soft_glow(canvas, width - 70, 98, 42)

        # 顶部：小标签 / 标题 / 大号更新时间 / 口径层级
        canvas.create_text(PAD + 8, 26, anchor="w", text="MARKET WATCH  ·  行情监控",
                           font=(FONT_CN, 9), fill=INK_FAINT)
        canvas.create_text(PAD + 8, 50, anchor="w", text="自选行情",
                           font=(FONT_CN, 15, "bold"), fill=INK)
        self.time_id = canvas.create_text(PAD + 6, 94, anchor="w", text="--:--:--",
                                          font=(FONT_NUM, 30, "bold"), fill=INK)
        self.sub_id = canvas.create_text(PAD + 8, 116, anchor="w",
                                         text=f"最后更新 · {len(self.symbols)} 只自选",
                                         font=(FONT_CN, 9), fill=INK_SOFT)

        # 右上角关闭按钮
        self._close_xy = (width - 32, 30)
        cx, cy = self._close_xy
        canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14,
                           fill=INK, outline=GLASS_EDGE, width=1)
        canvas.create_text(cx, cy - 1, text="×", fill="#ffffff",
                           font=(FONT_NUM, 15, "bold"))
        canvas.bind("<Button-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)

        # 行情胶囊列表（高度按股票数量自适应，3~5 只都合适）
        list_top = PAD + TOP_H
        hint_y = height - PAD - BOTTOM_H - HINT_H
        list_height = hint_y - list_top - 8
        row_height = list_height / len(self.symbols)
        card_height = min(row_height - 8, 82)
        compact = card_height < 66
        f_name = (FONT_CN, 11 if compact else 13, "bold")
        f_price = (FONT_NUM, 13 if compact else 16, "bold")
        f_day = (FONT_NUM, 10 if compact else 12, "bold")
        f_small = (FONT_CN, 8 if compact else 9)
        x1, x2 = PAD + 6, width - PAD - 6

        for index, sym in enumerate(self.symbols):
            y1 = list_top + index * row_height + (row_height - card_height) / 2
            y2 = y1 + card_height
            bg = _round_rect(canvas, x1, y1, x2, y2, 16,
                             fill=GLASS_HI, outline=GLASS_EDGE, width=1)
            r1 = y1 + card_height * 0.30
            r2 = y1 + card_height * 0.56
            r3 = y1 + card_height * 0.83
            items = {
                "bg": bg,
                "name": canvas.create_text(x1 + 16, r1, anchor="w", text="—",
                                           font=f_name, fill=INK),
                "price": canvas.create_text(x2 - 16, r1, anchor="e", text="—",
                                            font=f_price, fill=INK),
                "code": canvas.create_text(x1 + 16, r2, anchor="w", text=sym,
                                           font=f_small, fill=INK_FAINT),
                "day": canvas.create_text(x2 - 16, r2, anchor="e", text="—",
                                          font=f_day, fill=INK),
                "status": canvas.create_text(x1 + 16, r3, anchor="w", text="等待数据",
                                             font=f_small, fill=INK_FAINT),
                "min1": canvas.create_text(x2 - 16, r3, anchor="e", text="—",
                                           font=f_small, fill=INK_FAINT),
                # 复用第三行中部空白，不新增第四行（3~5 只都不挤压）
                "macd": canvas.create_text(MACD_BADGE_X, r3, anchor="center",
                                           text="", font=(FONT_CN, 8, "bold"),
                                           fill=INK_FAINT),
            }
            self.row_items[sym] = items

        # 计算基准说明（需求要求明确口径）
        canvas.create_text(width / 2, hint_y + HINT_H / 2, anchor="center",
                           text=HINT_TEXT, font=(FONT_CN, 8), fill=INK_FAINT)

        # 底部状态胶囊
        by1 = height - PAD - BOTTOM_H
        by2 = height - PAD
        _round_rect(canvas, PAD, by1, width - PAD, by2, 18,
                    fill=GLASS_HI, outline=GLASS_EDGE, width=1)
        self.status_id = canvas.create_text(PAD + 16, by1 + BOTTOM_H * 0.38,
                                            anchor="w", text="连接: 等待数据",
                                            font=(FONT_CN, 10, "bold"), fill=INK)
        self.meta_id = canvas.create_text(width - PAD - 16, by1 + BOTTOM_H * 0.38,
                                          anchor="e", text="—",
                                          font=(FONT_CN, 9), fill=INK_SOFT)

        self._refresh_table()
        self._refresh_status()

    def _on_press(self, event) -> None:
        cx, cy = self._close_xy
        if (event.x - cx) ** 2 + (event.y - cy) ** 2 <= 20 ** 2:
            self.on_close()
            return
        self._drag_origin = (event.x_root, event.y_root,
                             self.root.winfo_x(), self.root.winfo_y())

    def _on_drag(self, event) -> None:
        """无边框窗口自行实现拖动（用屏幕坐标，避免随窗口移动产生抖动）。"""
        if not self._drag_origin:
            return
        rx, ry, wx, wy = self._drag_origin
        self.root.geometry(f"+{wx + event.x_root - rx}+{wy + event.y_root - ry}")

    @staticmethod
    def _color(value) -> str:
        if value is None or value == 0:
            return INK
        return UP if value > 0 else DOWN

    @staticmethod
    def _short(text, limit: int = 36) -> str:
        text = str(text).replace("\n", " ")
        return text if len(text) <= limit else text[:limit - 1] + "…"

    def _refresh_table(self) -> None:
        for sym in self.symbols:
            state = self.states[sym]
            items = self.row_items[sym]
            dim = state["dim"]
            main_fg = DIM if dim else INK
            faint_fg = DIM if dim else INK_FAINT
            self.canvas.itemconfigure(items["bg"],
                                      fill=GLASS_DIM if dim else GLASS_HI)
            self.canvas.itemconfigure(items["name"], text=state["name"], fill=main_fg)
            self.canvas.itemconfigure(items["price"],
                                      text=core.fmt_price(state["price"]), fill=main_fg)
            self.canvas.itemconfigure(items["code"], text=sym, fill=faint_fg)
            self.canvas.itemconfigure(items["day"], text=core.fmt_pct(state["day"]),
                                      fill=DIM if dim else self._color(state["day"]))
            self.canvas.itemconfigure(items["min1"],
                                      text=f"近1分 {state['min1_text']}",
                                      fill=(DIM if dim else
                                            INK_FAINT if state["min1"] is None
                                            else self._color(state["min1"])))
            self.canvas.itemconfigure(items["status"],
                                      text=self._short(state["status"], 10),
                                      fill=(DIM if dim else
                                            WARN if state["warn"] else INK_FAINT))
            badge, badge_color = self.macd_states.get(sym, ("", INK_FAINT))
            self.canvas.itemconfigure(items["macd"], text=badge,
                                      fill=DIM if dim else badge_color)

    def _refresh_status(self) -> None:
        color = WARN if self.connection.startswith(("失败", "延迟", "界面异常")) else INK
        note = f" | {self.config_note}" if self.config_note else ""
        self.canvas.itemconfigure(
            self.status_id,
            text=f"连接: {self._short(self.connection + note, 24)}", fill=color)
        last = (self.last_success.strftime("%H:%M:%S")
                if self.last_success else "--:--:--")
        self.canvas.itemconfigure(self.time_id, text=last)
        source = SOURCE_LABEL.get(self.source_in_use, self.source_in_use)
        self.canvas.itemconfigure(
            self.meta_id,
            text=f"{self.cfg['refresh_interval_sec']}s · {source} · {self.session}")
        if self.cfg["macd"]["enabled"]:
            ready = sum(1 for sym in self.symbols
                        if self.macd_engine.rt.get(sym)
                        and self.macd_engine.rt[sym].warmup_ok)
            macd_note = f" · 1mMACD {ready}/{len(self.symbols)} 就绪"
        else:
            macd_note = " · 1mMACD 停用"
        self.canvas.itemconfigure(
            self.sub_id,
            text=self._short(f"最后更新 · {len(self.symbols)} 只自选{macd_note}", 30))

    # ------------------------------------------------------------- 后台取数
    def _fetch_loop(self) -> None:
        """首轮立即取数，之后按固定周期；Event.wait 可被关闭动作立即唤醒。

        周期固定为 refresh_interval_sec（把本轮取数耗时从下一轮等待里扣掉），
        否则实际周期会是「间隔 + 取数耗时」，超时切源时可能长到让
        DayChangeGate 的连续观测被判为中断，前置窗口永远累积不满。

        fetch_only_in_session=true 且为真实数据源时，非盘中一律跳过 HTTP：
        只投递一个轻量状态串让界面显示「暂停取数」，不伪造行情批次、
        不推高 tick（mock 场景的序列推进不受影响）。
        """
        interval = self.cfg["refresh_interval_sec"]
        tick = 0
        next_wait = 0.0                    # 首轮立即取数
        while not self.stop_event.wait(next_wait):
            next_wait = interval           # 成功取数后按耗时修正
            if self.stop_event.is_set():
                break
            now = core.beijing_now()
            if (self.cfg.get("fetch_only_in_session", True)
                    and self.cfg["data_source"] != "mock"
                    and core.session_of(now) != "盘中"):
                self.result_queue.put("paused:" + core.session_of(now))
                continue
            tick += 1
            started = time.monotonic()
            self.result_queue.put("updating")
            try:
                batch = core.fetch_quotes(self.symbols, self.cfg, tick)
                core.LOGGER.debug("第 %d 轮取数完成，耗时 %.2fs，源=%s",
                                  tick, time.monotonic() - started, batch.source)
            except Exception as exc:                       # 线程内异常不终止循环
                core.LOGGER.exception("取数线程异常")
                batch = core.Batch(core.beijing_now(), self.cfg["data_source"], {},
                                   {"__all__": f"{type(exc).__name__}: {exc}"})
            if self.stop_event.is_set():
                break
            self.result_queue.put(batch)
            elapsed = time.monotonic() - started
            next_wait = max(0.5, interval - elapsed)

    def _poll_queue(self) -> None:
        if self.closed or self.stop_event.is_set():
            return
        try:
            while True:
                item = self.result_queue.get_nowait()
                if isinstance(item, str) and item == "updating":
                    self.connection = "更新中"
                    self._refresh_status()
                elif isinstance(item, str) and item.startswith("paused:"):
                    label = item.split(":", 1)[1]
                    self.session = label
                    self.connection = f"{label}·暂停取数"
                    # 保留上一轮价格与涨跌颜色，只把行内状态改成明确的「已暂停」
                    for state in self.states.values():
                        state["status"] = f"{label}·暂停"
                    self._refresh_table()
                    self._refresh_status()
                elif isinstance(item, core.Batch):
                    self._apply_batch(item)
                elif isinstance(item, core.MacdFrame):
                    self._apply_macd_frame(item)
                else:
                    core.LOGGER.warning("忽略未知队列项: %r", type(item))
        except queue.Empty:
            pass
        except Exception:
            core.LOGGER.exception("处理行情批次异常")
        if not self.closed:
            self.after_id = self.root.after(100, self._poll_queue)

    # --------------------------------------------------------- 分钟线取数
    def _macd_fetch_loop(self) -> None:
        """独立 daemon：每分钟边界后拉一次分钟线，节奏与 5 秒快照完全解耦。"""
        macd = self.cfg["macd"]
        if not macd["enabled"]:
            return
        is_mock = self.cfg["data_source"] == "mock"
        cursor = {}
        tick = 0
        while not self.stop_event.is_set():
            now = core.beijing_now()
            if is_mock:
                wait_sec = max(1, macd["mock_step_sec"])
            else:
                wait_sec = max(1, min((60 - now.second) + macd["poll_delay_sec"], 90))
            if self.stop_event.wait(wait_sec):
                break
            now = core.beijing_now()
            if not is_mock and not core.macd_http_allowed(now, self.cfg):
                continue   # 默认严格盘中才请求；fetch_only_in_session=false 才允许盘后复盘
            tick += 1
            try:
                if is_mock:
                    per, session = {}, "盘中"
                    base_day = core.beijing_now().date().isoformat()
                    labels = core._mock_minute_labels(tick)
                    for index, sym in enumerate(self.symbols):
                        if index == 0:
                            fr = core.mock_minute_frame(sym, self.cfg, tick)
                            per[sym] = fr.per_symbol[sym]
                            session = fr.session
                        else:
                            # 其余股票横盘，不产生交叉，避免演示时刷屏
                            flat = tuple(core.MinuteBar(base_day, labels[i], 100.0)
                                         for i in range(tick))
                            per[sym] = core.MacdSymbolFrame(
                                sym, True, macd["primary_source"], base_day,
                                flat, labels[tick - 1])
                    frame = core.MacdFrame(core.beijing_now(),
                                           macd["primary_source"], per, session)
                else:
                    frame = core.fetch_macd_frame(
                        self.symbols, self.cfg, sleep_fn=self.stop_event.wait)
                    filtered = {}
                    for sym, sf in frame.per_symbol.items():
                        deliver, new_key = core.macd_cursor_advance(cursor, sym, sf)
                        if not deliver:
                            continue
                        if new_key is not None:
                            cursor[sym] = new_key   # 失败帧只投递、不写游标
                        filtered[sym] = sf
                    if not filtered:
                        continue
                    frame = core.MacdFrame(frame.fetch_ts, frame.source, filtered)
            except Exception:
                core.LOGGER.exception("分钟线取数线程异常")
                continue
            if self.stop_event.is_set():
                break
            self.result_queue.put(frame)

    def _apply_macd_frame(self, frame: core.MacdFrame) -> None:
        """主线程消费分钟线帧；只写角标/汇总/提醒，绝不回写快照连接状态。"""
        macd = self.cfg["macd"]
        if not macd["enabled"] or self.closed:
            return
        now = core.beijing_now()
        if (now - frame.fetch_ts).total_seconds() > macd["max_frame_age_sec"]:
            return
        if frame.session:
            session_ok = frame.session == "盘中"
            static_ok = False
        else:
            session = core.session_of(now)
            session_ok = (session == "盘中"
                          and core.macd_in_session_windows(now, self.cfg))
            static_ok = session != "盘中"

        # on_new_day 清的是全局状态，必须每帧只调用一次、且取帧内最新交易日：
        # 放在逐股循环里会被停牌股回吐的前一日序列反复触发，清空去重与限流状态。
        ok_dates = [sf.trade_date for sf in frame.per_symbol.values()
                    if sf.ok and sf.trade_date]
        if ok_dates:
            self.macd_engine.on_new_day(max(ok_dates))

        emitted = []
        for sym in self.symbols:
            sf = frame.per_symbol.get(sym)
            if sf is None:
                continue
            if sf.ok:
                points = core.compute_macd(sf.bars, self.cfg)
                crosses = core.detect_cross(points, self.cfg, sym, sf.trade_date)
            else:
                points, crosses = [], []
            emitted.extend(self.macd_engine.process(
                sym, sf, points, crosses, now, self.states[sym],
                session_ok=session_ok, static_ok=static_ok))
            badge = self.macd_engine.rt[sym].badge
            color = (WARN if not sf.ok else
                     UP if badge == core.BADGE_GOLDEN else
                     DOWN if badge == core.BADGE_DEATH else INK_FAINT)
            self.macd_states[sym] = (badge if macd["show_badge"] else "", color)
        for view in emitted:
            self._enqueue_alert(view)
        self._refresh_table()
        self._refresh_status()
        self._show_next_alert()

    # ------------------------------------------------------------- 数据应用
    def _apply_batch(self, batch: core.Batch) -> None:
        now = core.beijing_now()
        self.source_in_use = batch.source
        quote_times = [q.quote_ts for q in batch.quotes.values()]
        newest_ts = max(quote_times) if quote_times else None
        self.session = ("模拟" if batch.source == "mock"
                        else core.session_of(now, newest_ts))

        all_error = batch.errors.get("__all__")
        if all_error:
            age = ((now - self.last_success).total_seconds()
                   if self.last_success else None)
            detail = self._short(all_error, 60)
            if age is not None and age > self.cfg["stale_threshold_sec"]:
                self.connection = f"延迟(距上次成功 {int(age)}s): {detail}"
            else:
                self.connection = f"失败: {detail}"
            # 失败时保留上一轮数值、整行置灰，绝不清零、绝不冒充实时
            for sym in self.symbols:
                state = self.states[sym]
                state["dim"] = True
                state["warn"] = True
                state["status"] = "失败·保留上次值"
            self._refresh_table()
            self._refresh_status()
            return

        self.last_success = now
        if newest_ts is not None:
            if self.last_quote_ts == newest_ts:
                self.same_ts_count += 1
            else:
                self.same_ts_count = 0
            self.last_quote_ts = newest_ts

        alerts = []
        for sym in self.symbols:
            state = self.states[sym]
            if sym in batch.errors:
                state["dim"] = True
                state["warn"] = True
                state["status"] = self._short(batch.errors[sym], 24)
                continue

            quote = batch.quotes.get(sym)
            if quote is None:
                state["dim"] = True
                state["warn"] = True
                state["status"] = "无数据"
                continue

            reset_reason = self.snapshots.append(sym, quote, now)
            if reset_reason:
                self.engine.reset_symbol(sym)
                # 单向联动：除权/跨日会让 EMA 断档，MACD 计算态也需重置
                self.macd_engine.reset_symbol(sym, reset_reason)
                core.LOGGER.info("%s %s", sym, reset_reason)

            day_pct = core.compute_day_pct(quote.price, quote.prev_close)
            ref_price = self.snapshots.ref_price(sym, now)
            min1_pct = core.compute_min1_pct(quote.price, ref_price)
            # 必须在 evaluate 之前取：evaluate 内部会记录本轮观测并重置计时
            precheck_left = self.engine.day_precheck_remaining(sym, now)
            alert = self.engine.evaluate(sym, quote.name or self.names[sym],
                                         quote.price, day_pct, min1_pct, now)
            if alert:
                alerts.append(alert)

            state.update({
                "name": quote.name or self.names[sym],
                "price": quote.price,
                "day": day_pct,
                "min1": min1_pct,
                "min1_text": core.fmt_pct(min1_pct, "计算中"),
                "dim": False,
                "warn": False,
            })

            problems = []
            severe = False
            if (self.session == "盘中"
                    and (now - quote.quote_ts).total_seconds() > self.cfg["quote_max_age_sec"]):
                problems.append("行情过旧")
                severe = True
            if quote.quote_ts.date() != now.date():
                problems.append("非今日")
                severe = True
            if reset_reason:
                problems.append("快照重置")
            if self.session not in ("盘中", "模拟"):
                problems.append("静态行情")
            # 只有失败/延迟/异常才整行置灰；非交易时段的静态行情仍保留涨跌颜色
            if severe:
                state["dim"] = True
                state["warn"] = True
            state["status"] = "·".join(problems) if problems else "正常"
            # 当日涨幅已站上阈值、但「前 N 秒均在下方」尚未满足时不弹窗；
            # 在原有 status 位置给出倒计时，避免看起来像告警漏发。
            if (state["status"] == "正常" and precheck_left > 0
                    and day_pct is not None
                    and day_pct > self.cfg["day_change_threshold_pct"]
                    and not self.engine.locked.get((sym, "day"), False)):
                left = int(precheck_left)
                state["status"] = f"待确认{left // 60}:{left % 60:02d}"

        self.connection = self._global_status(now, batch)
        for alert in alerts:
            self._enqueue_alert(core.alert_to_view(alert))
        self._refresh_table()
        self._refresh_status()
        self._show_next_alert()

    def _global_status(self, now, batch: core.Batch) -> str:
        if batch.source == "mock":
            return "正常(模拟数据)"
        issues = []
        if (self.session == "盘中" and self.last_quote_ts is not None
                and (now - self.last_quote_ts).total_seconds() > self.cfg["quote_max_age_sec"]):
            issues.append("行情时间戳过旧")
        if self.session == "盘中" and self.same_ts_count >= 3:
            issues.append("行情未更新")
        if self.last_quote_ts is not None and self.last_quote_ts.date() != now.date():
            issues.append("行情非今日")
        if issues:
            return "延迟(" + "、".join(issues) + ")"
        if self.session != "盘中":
            return f"非交易时段·{self.session}(静态行情)"
        if batch.used_fallback:
            return "正常(已切备用源)"
        if batch.errors:
            return "部分股票异常"
        return "正常"

    # --------------------------------------------------------------- 提醒窗
    def _enqueue_alert(self, view: core.SignalView) -> None:
        self.alert_queue.append(view)
        while len(self.alert_queue) > self.cfg["alert_max_queue"]:
            self.alert_queue.popleft()
            core.LOGGER.warning("提醒队列已满，丢弃最旧一条提醒")

    def _alert_geometry(self, slot: int) -> str:
        """提醒窗固定在屏幕右下角；多条时向上堆叠，始终贴右侧。"""
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(0, screen_w - ALERT_W - ALERT_MARGIN_X)
        y = max(8, screen_h - ALERT_H - ALERT_MARGIN_Y
                - slot * (ALERT_H + ALERT_GAP))
        return f"{ALERT_W}x{ALERT_H}+{x}+{y}"

    def _show_next_alert(self) -> None:
        if self.closed:
            return
        screen_h = self.root.winfo_screenheight()
        capacity = max(1, min(self.cfg["alert_max_visible"],
                              (screen_h - 80) // (ALERT_H + ALERT_GAP)))
        while self.alert_queue and self.open_alerts < capacity:
            self._make_alert_window(self.alert_queue.popleft())

    def _make_alert_window(self, view: core.SignalView) -> None:
        """右下角玻璃提醒卡：统一渲染涨幅告警与 MACD 状态提示，非模态不阻塞。"""
        slot = 0
        while slot in self.alert_slots:
            slot += 1
        self.alert_slots.add(slot)
        self.open_alerts += 1

        width, height = ALERT_W, ALERT_H
        win = tk.Toplevel(self.root)
        win.title("行情提醒")
        win.geometry(self._alert_geometry(slot))
        win.resizable(False, False)
        glass = self._apply_glass(win)
        win.attributes("-topmost", True)

        canvas = tk.Canvas(win, width=width, height=height,
                           bg=(KEY if glass else GLASS),
                           highlightthickness=0, bd=0)
        canvas.pack()
        _round_rect(canvas, 2, 2, width - 2, height - 2, 24,
                    fill=GLASS, outline=GLASS_EDGE, width=1.5)
        _soft_glow(canvas, width - 58, 70, 38)

        def text(x, y, value, font, fill, anchor="w"):
            return canvas.create_text(x, y, anchor=anchor, text=value,
                                      font=font, fill=fill)

        is_macd = view.kind == "macd"
        text(20, 28, "ALERT  ·  指标状态提示" if is_macd else "ALERT  ·  行情提醒",
             (FONT_CN, 9), INK_FAINT)
        text(20, 56, self._short(f"{view.name}  ({view.symbol})", 26),
             (FONT_CN, 13, "bold"), INK)
        text(20, 96, core.fmt_price(view.price), (FONT_NUM, 26, "bold"), INK)
        canvas.create_line(20, 122, width - 20, 122, fill="#ded5c8")

        text(20, 146, "当日涨跌幅", (FONT_CN, 9), INK_SOFT)
        text(width - 20, 146, core.fmt_pct(view.day_pct),
             (FONT_NUM, 12, "bold"), self._color(view.day_pct), anchor="e")
        text(20, 170, "近1分钟涨幅", (FONT_CN, 9), INK_SOFT)
        text(width - 20, 170, core.fmt_pct(view.min1_pct, "计算中"),
             (FONT_NUM, 12, "bold"),
             INK_FAINT if view.min1_pct is None else self._color(view.min1_pct),
             anchor="e")

        if is_macd:
            signal_color = UP if view.direction == "golden" else DOWN
            cross_desc = ("DIF 上穿 DEA · 0 轴上方" if view.direction == "golden"
                          else "DIF 下穿 DEA · 0 轴下方")

            def num3(value):
                return "—" if value is None else f"{value:+.3f}"

            text(20, 200, view.conditions, (FONT_CN, 11, "bold"), signal_color)
            text(20, 218, cross_desc, (FONT_CN, 9), INK_SOFT)
            text(20, 244, (f"DIF {num3(view.dif)}   DEA {num3(view.dea)}   "
                           f"柱 {num3(view.bar)}"), (FONT_NUM, 9), INK)
            kline = (f"{view.kline_minute[:2]}:{view.kline_minute[2:]}"
                     if len(view.kline_minute) == 4 else view.kline_minute)
            confirm_short = view.confirm_note.split(" · ")[0]
            text(20, 262, self._short(
                f"K线分钟 {kline}（已收盘）· {confirm_short}", 36),
                (FONT_CN, 8), INK_FAINT)
            text(20, 286, f"触发时间 {view.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}",
                 (FONT_NUM, 9), INK)
        else:
            text(20, 200, "触发条件", (FONT_CN, 9), INK_SOFT)
            text(20, 218, self._short(view.conditions, 34), (FONT_CN, 9), ACCENT)
            text(20, 244, "触发时间", (FONT_CN, 9), INK_SOFT)
            text(20, 262, view.triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
                 (FONT_NUM, 10), INK)
            source = SOURCE_LABEL.get(self.source_in_use, self.source_in_use)
            text(20, 286, f"数据源 {source} · {self.session}",
                 (FONT_CN, 8), INK_FAINT)

        # 免责全文按 " · " 分段折行：40 个中文字在 340px 内单行会被裁切，
        # 且不能用 _short 省略号丢掉「不构成投资建议」。
        for i, line in enumerate(_wrap_disclaimer(core.MACD_DISCLAIMER)):
            text(20, 306 + i * 16, line, (FONT_CN, 8), INK_FAINT)
        auto_close = self.cfg["alert_auto_close_sec"]
        text(20, 340, (f"{auto_close}s 后自动关闭" if auto_close > 0
                       else "需手动关闭"), (FONT_CN, 8), INK_FAINT)
        close_xy = (width - 34, 344)
        cx, cy = close_xy
        canvas.create_oval(cx - 15, cy - 15, cx + 15, cy + 15,
                           fill=INK, outline=GLASS_EDGE, width=1)
        canvas.create_text(cx, cy - 1, text="×", fill="#ffffff",
                           font=(FONT_NUM, 15, "bold"))

        done = [False]

        def close():
            if done[0]:
                return
            done[0] = True
            self.alert_slots.discard(slot)
            try:
                if win.winfo_exists():
                    win.destroy()
            except tk.TclError:
                pass
            self.open_alerts = max(0, self.open_alerts - 1)
            self._show_next_alert()

        def on_click(event):
            if (event.x - cx) ** 2 + (event.y - cy) ** 2 <= 20 ** 2:
                close()

        canvas.bind("<Button-1>", on_click)
        win.protocol("WM_DELETE_WINDOW", close)
        if auto_close > 0:
            win.after(auto_close * 1000, close)
        # 焦点交还主窗口（不用 focus_force，避免把主窗口强行提到前台）
        def return_focus():
            try:
                if self.root.winfo_exists():
                    self.root.focus_set()
            except tk.TclError:
                pass
        self.root.after(1, return_focus)

    # --------------------------------------------------------------- 生命周期
    def _on_tk_error(self, exc, value, tb) -> None:
        core.LOGGER.error("界面回调异常", exc_info=(exc, value, tb))
        try:
            self.connection = "界面异常(详见日志)"
            self._refresh_status()
        except Exception:
            pass

    def on_close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.after_id:
            try:
                self.root.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.macd_thread.is_alive():
            self.macd_thread.join(timeout=1)
        core.LOGGER.info("行情监控已关闭")
        try:
            self.root.destroy()          # Toplevel 提醒窗随之销毁
        except tk.TclError:
            pass


def main() -> None:
    cfg, note = core.load_config(CONFIG_PATH)
    setup_logging(cfg)
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法创建窗口: {exc}", file=sys.stderr)
        sys.exit(2)
    MonitorApp(root, cfg, note)
    root.mainloop()


if __name__ == "__main__":
    main()
