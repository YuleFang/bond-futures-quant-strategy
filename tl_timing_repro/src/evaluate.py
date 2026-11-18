"""评估层：绩效指标、与原文数值差异表、交付前自检。"""

from __future__ import annotations

import importlib
import json
import os

import numpy as np
import pandas as pd

from .models import binary12_metrics


def compute_metrics(nav_df: pd.DataFrame, trades: pd.DataFrame, cfg: dict) -> dict:
    m = cfg["metrics"]
    af = int(m["annual_factor"])
    rf = float(m["rf"])
    nav = nav_df["nav"].to_numpy(dtype=float)
    dr = nav_df["daily_ret"].to_numpy(dtype=float)
    n = len(nav)
    total = float(nav[-1] / nav[0] - 1.0)
    annual = float((nav[-1] / nav[0]) ** (af / max(n, 1)) - 1.0) if n > 1 else 0.0
    vol = float(np.std(dr, ddof=1) * np.sqrt(af)) if n > 2 else 0.0
    sharpe = float((np.mean(dr) - rf / af) / np.std(dr, ddof=1) * np.sqrt(af)) if np.std(dr, ddof=1) > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = float(abs(np.min(nav / peak - 1.0)))  # 回撤幅度，取正值，与原文口径一致
    win_rate = float((trades["ret"] > 0).mean()) if len(trades) else 0.0
    bench = nav_df["benchmark_nav"].to_numpy(dtype=float)
    return {
        "sample_start": str(nav_df["date"].iloc[0].date()),
        "sample_end": str(nav_df["date"].iloc[-1].date()),
        "n_trading_days": int(n),
        "cumulative_return": total,
        "annual_return": annual,
        "annual_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "n_trades": int(len(trades)),
        "benchmark_cumulative_return": float(bench[-1] / bench[0] - 1.0),
        "cost_assumption": cfg["backtest"]["costs"],
        "rf": rf,
        "annual_factor": af,
        "win_rate_denominator": m["win_rate_denominator"],
    }


def comparison_table(metrics: dict, cls: dict, cfg: dict) -> pd.DataFrame:
    ref = cfg["paper_reference"]
    rows = [
        ("年化收益", metrics["annual_return"], ref["annual_return"], "pct"),
        ("夏普比", metrics["sharpe"], ref["sharpe"], "num"),
        ("最大回撤", metrics["max_drawdown"], ref["max_drawdown"], "pct"),
        ("交易胜率", metrics["win_rate"], ref["win_rate"], "pct"),
        ("Precision(类1+2)", cls["precision"], ref["precision_12"], "pct"),
        ("Recall(类1+2)", cls["recall"], ref["recall_12"], "pct"),
        ("F1(类1+2)", cls["f1"], ref["f1_12"], "pct"),
    ]
    out = []
    for name, ours, paper, unit in rows:
        diff = ours - paper
        out.append({
            "指标": name,
            "本次复现": round(ours, 4),
            "原文": round(paper, 4),
            "差异": round(diff, 4),
            "单位": unit,
        })
    return pd.DataFrame(out)


def delivery_self_check(cfg: dict, no_lookahead: dict, workdir: str) -> dict:
    """交付前自检：无未来函数、成本参数一致、依赖可导入、缓存完整。"""
    checks = []
    checks.append({
        "item": "无未来函数（因子扰动测试）",
        "passed": bool(no_lookahead["passed"]),
        "detail": f"max_abs_diff={no_lookahead['max_abs_diff']:.3e}, "
                  f"probe={no_lookahead['probe_date']}, rows={no_lookahead['rows_checked']}",
    })
    costs = cfg["backtest"]["costs"]
    checks.append({
        "item": "成本参数与配置一致（零成本口径）",
        "passed": all(float(v) == 0.0 for v in costs.values()),
        "detail": json.dumps(costs, ensure_ascii=False),
    })
    core = ["pandas", "numpy", "yaml"]
    optional = ["scipy", "sklearn", "xgboost", "lightgbm", "catboost", "statsmodels"]
    missing_core, missing_opt = [], []
    for d in core + optional:
        try:
            importlib.import_module(d)
        except Exception:
            (missing_core if d in core else missing_opt).append(d)
    checks.append({
        "item": "核心依赖可导入",
        "passed": len(missing_core) == 0,
        "detail": "全部可用" if not missing_core else f"缺失: {missing_core}",
    })
    checks.append({
        "item": "模型/统计后端状态（信息项，不阻断）",
        "passed": True,
        "detail": "全部可用" if not missing_opt else (
            f"缺失 {missing_opt}；预处理与模型已启用纯 numpy 降级实现，"
            "该降级须在报告中登记为偏离"
        ),
    })
    required = [
        cfg["data"]["segment_a"]["file"], cfg["data"]["segment_b"]["file"],
        *[v["file"] for v in cfg["data"]["edb"].values()],
    ]
    absent = [f for f in required if not os.path.exists(os.path.join(workdir, cfg["data"]["cache_dir"], f))]
    checks.append({
        "item": "Wind 缓存文件完整",
        "passed": len(absent) == 0,
        "detail": "全部存在" if not absent else f"缺失: {absent}",
    })
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks)}


def format_metrics_for_report(metrics: dict) -> str:
    return (
        f"年化 {metrics['annual_return']:.2%} / 夏普 {metrics['sharpe']:.2f} / "
        f"最大回撤 {metrics['max_drawdown']:.2%} / 胜率 {metrics['win_rate']:.2%} / "
        f"交易 {metrics['n_trades']} 笔"
    )
