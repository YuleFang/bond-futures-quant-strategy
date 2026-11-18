"""参数敏感性与样本内外对比。

对 holding_days / mad_n / rolling_window_daily / seed 做单因子扰动，
每次只改一个参数，其余保持基准配置；输出各情形的绩效指标供过拟合判断。
"""

from __future__ import annotations

import copy

import pandas as pd


def _override(cfg: dict, path: tuple, value) -> dict:
    c = copy.deepcopy(cfg)
    node = c
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return c


def sweep(run_once, cfg: dict) -> pd.DataFrame:
    """run_once(cfg) -> dict（至少含 metrics 字段）。"""
    spec = cfg["sensitivity"]
    jobs = []
    for v in spec["holding_days"]:
        jobs.append(("holding_days", v, ("backtest", "holding_days"), v))
    for v in spec["mad_n"]:
        jobs.append(("mad_n", v, ("preprocess", "mad_n"), v))
    for v in spec["rolling_window_daily"]:
        jobs.append(("rolling_window_daily", v, ("preprocess", "rolling_window_daily"), v))
    for v in spec["seeds"]:
        jobs.append(("seed", v, ("project", "seed"), v))

    rows = []
    for name, value, path, _ in jobs:
        c = _override(cfg, path, value)
        try:
            res = run_once(c)
            m = res["metrics"]
            rows.append({
                "param": name,
                "value": value,
                "annual_return": round(m["annual_return"], 6),
                "sharpe": round(m["sharpe"], 6),
                "max_drawdown": round(m["max_drawdown"], 6),
                "win_rate": round(m["win_rate"], 6),
                "n_trades": m["n_trades"],
                "f1_12": round(res["classification"]["f1"], 6),
                "status": "ok",
            })
        except Exception as exc:
            rows.append({
                "param": name, "value": value, "status": f"失败: {type(exc).__name__}: {exc}",
            })
    return pd.DataFrame(rows)
