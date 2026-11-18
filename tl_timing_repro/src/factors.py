"""因子与标签构造。

原则
----
* 所有因子只用 ``<= t`` 的信息，全部由可显式写出的公式构造；不造原文未定义的
  非线性交叉项。
* 技术面因子在段内各自计算，**不跨段拼接**（段 A 用 T.CFE，段 B 用 TL.CFE）。
* EDB 指标按交易日对齐后只用已发布观测做前向填充。
* 标签：``r_t = close[t+5]/close[t] - 1``；扩展窗口分位数只用 t 之前已完全
  实现的 5 日收益，避免未来函数。
* 方向映射：利率下行 ↔ 期货上涨，故原文利率变动的低分位对应期货收益的高分位：
  ``r`` 分位 >0.80 -> 类别 2；0.60~0.80 -> 类别 1；<0.60 -> 类别 0
  （含原文未定义的「上行」样本，已确认并入类别 0）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_fetch import align_edb_to_trading_days


def build_price_factors(px: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """由 OHLCV 构造技术面因子，index 为日期。"""
    f = pd.DataFrame(index=pd.DatetimeIndex(px["date"]))
    close = px["close"].astype(float).to_numpy()
    high = px["high"].astype(float).to_numpy()
    low = px["low"].astype(float).to_numpy()
    opn = px["open"].astype(float).to_numpy()
    vol = px["volume"].astype(float).to_numpy()
    s = pd.Series(close, index=f.index)

    for w in cfg["factors"]["price_windows"]:
        f[f"ret_{w}"] = s / s.shift(w) - 1.0
    r1 = s / s.shift(1) - 1.0
    for w in cfg["factors"]["vol_windows"]:
        f[f"vol_{w}"] = r1.rolling(w).std()
    for w in cfg["factors"]["ma_windows"]:
        f[f"ma_bias_{w}"] = s / s.rolling(w).mean() - 1.0

    # RSI
    n = int(cfg["factors"]["rsi_window"])
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    f[f"rsi_{n}"] = 100 - 100 / (1 + rs)

    amplitude = (high - low) / np.where(close == 0, np.nan, close)
    f["amplitude"] = amplitude
    f["range_mean_5"] = pd.Series(amplitude, index=f.index).rolling(5).mean()
    denom = np.where(high == low, np.nan, high - low)
    f["close_position"] = (close - low) / denom
    f["gap"] = opn / s.shift(1) - 1.0
    f["log_volume"] = np.log(np.where(vol <= 0, np.nan, vol))
    f["volume_chg_5"] = pd.Series(vol, index=f.index) / pd.Series(vol, index=f.index).shift(5) - 1.0
    f["momentum_accel"] = f["ret_5"] - f["ret_10"] if "ret_10" in f else s / s.shift(5) - s / s.shift(10)
    f["skew_20"] = r1.rolling(20).skew()
    f["volume_ma_bias_20"] = (
        pd.Series(vol, index=f.index) / pd.Series(vol, index=f.index).rolling(20).mean() - 1.0
    )
    return f.astype(float)


def build_edb_factors(px: pd.DataFrame, edb: dict, cfg: dict) -> pd.DataFrame:
    """由 EDB 指标构造流动性 / 高频基本面因子。"""
    aligned = align_edb_to_trading_days(edb, px["date"], cfg)
    f = pd.DataFrame(index=aligned.index)
    for code in cfg["data"]["edb"]:
        f[code] = aligned[code]
    for w in cfg["factors"]["edb_chg_windows"]:
        for code in ("M1006337", "M0041652", "S5914515"):
            f[f"{code}_chg_{w}"] = aligned[code] / aligned[code].shift(w) - 1.0
        f[f"S0114089_chg_{w}"] = aligned["S0114089"] / aligned["S0114089"].shift(w) - 1.0
    f["dr007_r001_spread"] = aligned["M1006337"] - aligned["M0041652"]
    f["dr007_ma_bias_20"] = aligned["M1006337"] / aligned["M1006337"].rolling(20).mean() - 1.0
    for code in cfg["factors"]["log_transform"]:
        col = aligned[code]
        f[f"log_{code}"] = np.log(np.where(col <= 0, np.nan, col))
    return f.astype(float)


def build_features(px: pd.DataFrame, edb: dict, cfg: dict) -> pd.DataFrame:
    pf = build_price_factors(px, cfg)
    ef = build_edb_factors(px, edb, cfg)
    out = pd.concat([pf, ef], axis=1)
    out.index.name = "date"
    return out


def build_labels(px: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """构造三分类标签与前向收益。返回 index=date 的 DataFrame。"""
    lcfg = cfg["label"]
    h = int(lcfg["horizon"])
    close = px["close"].astype(float).to_numpy()
    dates = pd.DatetimeIndex(px["date"])
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > h:
        fwd[: n - h] = close[h:] / close[: n - h] - 1.0

    q_low = float(lcfg["q_small_down"])
    q_high = float(lcfg["q_large_down"])
    min_hist = int(lcfg["min_history"])
    q60 = np.full(n, np.nan)
    q80 = np.full(n, np.nan)
    # 在时点 i，只有窗口结束日 <= i 的 fwd（即索引 <= i-h）才已完全实现
    for i in range(n):
        end = i - h
        if end < 0:
            continue
        hist = fwd[: end + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_hist:
            q60[i], q80[i] = np.quantile(hist, [q_low, q_high])

    label = np.full(n, np.nan)
    valid = ~np.isnan(fwd) & ~np.isnan(q60) & ~np.isnan(q80)
    label[valid] = 0
    label[valid & (fwd > q60) & (fwd <= q80)] = 1
    label[valid & (fwd > q80)] = 2
    return pd.DataFrame(
        {"fwd_ret_5": fwd, "q60": q60, "q80": q80, "label": label}, index=dates
    )


def assert_no_lookahead(px: pd.DataFrame, edb: dict, cfg: dict, probe: int | None = None) -> dict:
    """无未来函数自检：扰动 probe 之后的收盘价，检查 probe 及之前的因子不变。"""
    n = len(px)
    probe = probe or int(n * 0.7)
    orig = build_features(px, edb, cfg)
    perturbed = px.copy()
    close = perturbed["close"].astype(float).to_numpy().copy()
    close[probe + 1 :] = close[probe + 1 :] * 1.37 + 3.21
    perturbed["close"] = close
    after = build_features(perturbed, edb, cfg)
    a = orig.iloc[: probe + 1].to_numpy(dtype=float)
    b = after.iloc[: probe + 1].to_numpy(dtype=float)
    max_diff = float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0
    result = {
        "probe_index": probe,
        "probe_date": str(px["date"].iloc[probe].date()),
        "rows_checked": probe + 1,
        "max_abs_diff": max_diff,
        "passed": bool(max_diff < 1e-10),
    }
    return result
