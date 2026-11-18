"""回测层：赔率视角的做多信号 -> TL 多头交易。

确认口径（贴原文档）：
* 信号类别 1 或 2 -> 买点；**信号当日收盘建仓**（已知轻微前视偏差，已登记）。
* 持有 5 个交易日后收盘平仓；持有期内再现买点 -> **重置**持有期，不加仓、
  不重复建仓。
* 100% 名义仓位，只做多、不做空、不加杠杆、不计保证金；成本 0/0/0。
* 基准 = 样本外区间 TL 买入持有。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(px: pd.DataFrame, signal: np.ndarray, cfg: dict) -> dict:
    """在给定价格序列上按信号交易。

    Parameters
    ----------
    px : 含 date / close 的 DataFrame（样本外区间，按日期升序）
    signal : 与 px 等长的预测类别数组（0/1/2，允许 NaN 视为 0）
    """
    b = cfg["backtest"]
    h = int(b["holding_days"])
    close = px["close"].astype(float).to_numpy()
    dates = pd.DatetimeIndex(px["date"])
    n = len(close)
    sig = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0).astype(int)

    position = np.zeros(n, dtype=int)   # 当日收盘后是否持仓
    trades = []
    in_pos = False
    entry_idx = None
    exit_idx = None

    for i in range(n):
        # 到期平仓（收盘）
        if in_pos and i >= exit_idx:
            trades.append({
                "entry_date": str(dates[entry_idx].date()),
                "exit_date": str(dates[i].date()),
                "entry_price": float(close[entry_idx]),
                "exit_price": float(close[i]),
                "ret": float(close[i] / close[entry_idx] - 1.0),
                "holding_days": int(i - entry_idx),
            })
            in_pos = False
            entry_idx = None
        # 信号触发
        if sig[i] in (1, 2):
            if not in_pos:
                in_pos = True
                entry_idx = i
                exit_idx = i + h
            elif b["reset_on_new_signal"]:
                exit_idx = i + h   # 重置，不加仓
        position[i] = int(in_pos)

    # 期末强制平仓（若仍持仓）
    if in_pos:
        last = n - 1
        trades.append({
            "entry_date": str(dates[entry_idx].date()),
            "exit_date": str(dates[last].date()),
            "entry_price": float(close[entry_idx]),
            "exit_price": float(close[last]),
            "ret": float(close[last] / close[entry_idx] - 1.0),
            "holding_days": int(last - entry_idx),
            "forced_close": True,
        })

    daily_ret = np.zeros(n)
    if n > 1:
        price_ret = close[1:] / close[:-1] - 1.0
        daily_ret[1:] = position[:-1] * price_ret   # 建仓当日不计收益
    cost = sum(float(v) for v in b["costs"].values())
    daily_ret = daily_ret - cost * (np.abs(np.diff(np.r_[0, position])) > 0).astype(float)

    nav = np.cumprod(1.0 + daily_ret)
    bench_nav = close / close[0]
    out = pd.DataFrame({
        "date": dates, "close": close, "signal": sig, "position": position,
        "daily_ret": daily_ret, "nav": nav, "benchmark_nav": bench_nav,
    })
    return {
        "nav": out,
        "trades": pd.DataFrame(trades),
        "n_signals": int((sig > 0).sum()),
        "n_trades": int(len(trades)),
        "exposure": float(position.mean()),
    }
