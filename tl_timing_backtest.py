"""
tl_timing_backtest.py
=====================================================================
《赔率视角下的30年国债择时模型》（浙商固收 / 覃汉研究笔记，2025-06-20）
交易逻辑的可执行复现 + 工程化改进。

---------------------------------------------------------------------
策略设计（结构化）
---------------------------------------------------------------------
1. 核心思路
   用基本面/资金面/曲线结构/情绪面/技术面因子，预测「未来 5 个交易日
   30 年期国债收益率的变动方向」，只做多（赔率视角）：模型判断利率下行
   → 30Y 国债期货 TL 上涨 → 买入 TL；判断震荡/上行 → 空仓。不做空。

2. 标签（三分类，原文口径）
   以 t 时点已实现的历史 5 日收益率变动的滚动分位数 q20 / q40 / q60
   为阈值，对 Δy = y30(t+5) - y30(t) 分类：
     类别 2 大幅下行 : Δy <= q20
     类别 1 小幅下行 : q20 < Δy <= q40
     类别 0 震荡/非买点 : q40 < Δy（含原文 40%~60% 震荡及 >60% 上行）
   原文用全样本历史分位数（含未来信息），本实现改为「截至 t 的滚动
   756 日分位数」，严格避免前视偏差。

3. 模型
   XGBoost / LightGBM / CatBoost 三基模型 + 动态 F1 加权集成。
   时间轴按自然年分块，每年前 80% 训练、后 20% 验证；训练集对类别
   1/2 上采样至类别均衡；验证集 F1 用于模型权重与概率阈值选择；
   样本外按滚动 120 日、滞后 5 日已实现标签动态更新权重。

4. 信号
   原文规则：三分类模型 argmax 输出为类别 1 或类别 2 → 买点。
   另设 p_long = P(类别1)+P(类别2) ≥ threshold 的概率阈值规则，
   仅用于参数敏感性对照；threshold 在验证集上以「最大化做多 F1 且
   信号率 ≤ 60%」选出，避免 F1 最大化退化为几乎全做多。

5. 交易规则
   开仓：t 日收盘后产生买点 → t+1 日开盘按开盘价买入 TL 主力连续。
   平仓：持有满 H=5 个交易日 → 次日开盘卖出；
         持有期内再次出现买点 → 持有期从该信号顺延 H 日（上限 20 日）。
   止盈止损（本实现的增强，原文无）：
         止损 = 入场价 ×(1 - SL)，以当日最低价触发；
         止盈 = 入场价 ×(1 + TP)，以当日最高价触发；
         同日同时触发时止损优先；跳空穿越按开盘价成交。
         基准情形 SL/TP 关闭，以贴合原文。
   仓位：单标的、满仓、1 倍价格指数化敞口，不考虑保证金/杠杆/合约乘数。
   成本：基准 0 费用 0 滑点（贴合原文），敏感性测试加入双边费用+滑点。

6. 回测区间与标的
   训练：2015-05-14 ~ 2024-05-14（因子与标签，收益率曲线可得的最长区间）
   样本外主区间：2024-05-15 ~ 2025-05-14（与原文一致）
   扩展样本外：2025-05-15 ~ 2026-11-11（稳健性检验）
   交易标的：TL.CFE 30 年期国债期货主力连续（公开数据用 TL0 连续合约）
   注意 TL 于 2023-04-21 上市，训练期的期货技术面因子为中性值 0。

7. 绩效指标
   累计/年化收益、夏普比率(252, 无风险利率0)、最大回撤、Calmar、
   交易胜率、交易次数、平均持仓天数、基准(TL 买入持有)对比，以及
   做多信号的 Precision / Recall / F1。

---------------------------------------------------------------------
运行
---------------------------------------------------------------------
   python tl_data.py                 # 首次取数并缓存（需 akshare）
   python tl_timing_backtest.py      # 训练 + 回测 + 敏感性 + 导出
   python tl_render_dashboard.py    # 渲染中文 HTML 仪表盘
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from export_results import export_results
from tl_data import DATA_END, DATA_START, FEATURE_CSV, PANEL_CSV, build_features, load_panel

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
TRAIN_START = "2015-05-14"
TRAIN_END = "2024-05-14"
OOS_START = "2024-05-15"
OOS_END = "2025-05-14"
EXT_START = "2025-05-15"
EXT_END = DATA_END
INITIAL_CASH = 1_000_000.0
SYMBOL = "TL.CFE"
SYMBOL_NAME = "30年期国债期货主力连续"
PREFIX = "tl_ensemble"
SEED = 20250620

FEATURE_EXCLUDE = {"date", "fwd_dy30_5", "q20", "q40", "q60", "label", "label_long",
                   "g_30年", "g_10年", "g_1年", "tl_open", "tl_high", "tl_low", "tl_close"}


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def make_models():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "XGB": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                             tree_method="hist", n_jobs=4, random_state=SEED,
                             objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                             verbosity=0),
        "LGB": LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                              n_jobs=4, random_state=SEED, verbose=-1),
        "CAT": CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05,
                                  loss_function="MultiClass", l2_leaf_reg=3.0,
                                  thread_count=4, random_seed=SEED, verbose=False,
                                  allow_writing_files=False),
    }


def upsample(X: pd.DataFrame, y: np.ndarray, rng: np.random.Generator):
    """训练集对类别 1 / 2 上采样至与类别 0 等量（原文 3.4.2）。"""
    parts = {c: np.flatnonzero(y == c) for c in (0, 1, 2)}
    target = len(parts[0])
    picks = [parts[0]]
    for c in (1, 2):
        src = parts[c]
        picks.append(rng.choice(src, size=target, replace=True) if len(src) else
                     np.array([], dtype=int))
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx].astype(int)


def _time_blocks(dates: pd.Series, start: str, end: str):
    """按自然年切块：每年前 80% 训练、后 20% 验证（原文 3.4.1）。"""
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    base = np.flatnonzero(mask.to_numpy())
    tr, va = [], []
    for _, grp in pd.Series(base, index=dates.iloc[base].dt.year.to_numpy()).groupby(level=0):
        arr = grp.to_numpy()
        cut = int(len(arr) * 0.8)
        tr.extend(arr[:cut]); va.extend(arr[cut:])
    return np.array(sorted(tr)), np.array(sorted(va))


def best_threshold(p_long: np.ndarray, y_true: np.ndarray,
                   max_signal_rate: float = 0.60) -> float:
    """验证集选概率阈值：最大化做多 F1，但约束信号率 ≤ 60%。

    仅最大化 F1 在正样本基准率约 50% 时会退化为「几乎全部预测做多」，
    使择时策略退化为买入持有；加入信号率上限可保留真实择时能力。
    该阈值只用于参数敏感性对照，主策略采用原文的 argmax 分类规则。
    """
    best, best_f1 = 0.5, -1.0
    for t in np.arange(0.10, 0.91, 0.02):
        pred = (p_long >= t).astype(int)
        if pred.sum() == 0 or pred.mean() > max_signal_rate:
            continue
        s = f1_score(y_true, pred, zero_division=0)
        if s > best_f1:
            best, best_f1 = float(t), s
    return best


def train_and_predict(feat: pd.DataFrame, verbose: bool = True):
    cols = [c for c in feat.columns if c.startswith("z_") and c not in FEATURE_EXCLUDE]
    X = feat[cols].astype(float)
    y = feat["label"].to_numpy()
    dates = feat["date"]

    tr_all = ((dates >= pd.Timestamp(TRAIN_START)) & (dates <= pd.Timestamp(TRAIN_END))
              & feat["label"].notna()).to_numpy()
    tr_idx, va_idx = _time_blocks(dates, TRAIN_START, TRAIN_END)
    tr_idx = np.array([i for i in tr_idx if tr_all[i]])
    va_idx = np.array([i for i in va_idx if tr_all[i]])

    Xtr, ytr = upsample(X.iloc[tr_idx].reset_index(drop=True),
                        y[tr_idx].astype(int), np.random.default_rng(SEED))
    # upsample 返回的位置索引需与 reset_index 后的 X 对齐
    Xva, yva = X.iloc[va_idx], y[va_idx].astype(int)
    yva_long = np.isin(yva, [1, 2]).astype(int)

    models = make_models()
    fitted, val_probs, val_f1, thresholds = {}, {}, {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        fitted[name] = mdl
        p = mdl.predict_proba(Xva)
        pl = p[:, 1] + p[:, 2]
        t = best_threshold(pl, yva_long)
        thresholds[name] = t
        val_probs[name] = pl
        val_f1[name] = f1_score(yva_long, (pl >= t).astype(int), zero_division=0)
        if verbose:
            print(f"[model] {name}: val F1={val_f1[name]:.4f}, threshold={t:.2f}, "
                  f"train={len(Xtr)}, val={len(Xva)}")

    # 预测范围：TRAIN_END 之后全部交易日（真实样本外，固定模型，不再训练）
    pred_mask = (dates > pd.Timestamp(TRAIN_END)).to_numpy()
    pred_idx = np.flatnonzero(pred_mask)
    proba = {}
    proba_full = {}
    for name, mdl in fitted.items():
        p = mdl.predict_proba(X.iloc[pred_idx])
        proba_full[name] = p
        proba[name] = p[:, 1] + p[:, 2]

    # 验证集 F1 权重（动态权重的初值）
    weights0 = {k: max(v, 1e-4) for k, v in val_f1.items()}
    total = sum(weights0.values())
    weights0 = {k: v / total for k, v in weights0.items()}

    # 集成阈值：验证集上最大化 F1
    p_ens_va = sum(weights0[k] * val_probs[k] for k in models)
    thr_ens = best_threshold(p_ens_va, yva_long)
    f1_ens_va = f1_score(yva_long, (p_ens_va >= thr_ens).astype(int), zero_division=0)
    if verbose:
        print(f"[model] ENS: val F1={f1_ens_va:.4f}, threshold={thr_ens:.2f}, "
              f"weights={ {k: round(v, 3) for k, v in weights0.items()} }")

    # 动态权重：对每个预测日，用截至 t-5 已实现标签的滚动 120 日 F1 加权
    label_long = feat["label_long"].to_numpy()
    pos_map = {int(idx_): p for p, idx_ in enumerate(pred_idx)}
    dynamic_w, ens_long, ens_full = [], [], []
    for j, i in enumerate(pred_idx):
        # 只使用「向前 5 个交易日已走完」的样本：pred_idx[j-5] 及更早
        realized = pred_idx[max(0, j - 120): max(0, j - 5)]
        realized = realized[~np.isnan(label_long[realized])]
        if len(realized) < 40:
            w = weights0
        else:
            yy = label_long[realized].astype(int)
            w = {}
            for name in models:
                rp = np.array([proba[name][pos_map[int(r)]] for r in realized])
                w[name] = f1_score(yy, (rp >= thresholds[name]).astype(int), zero_division=0) + 1e-4
            s = sum(w.values())
            w = {k: v / s for k, v in w.items()}
        dynamic_w.append(w)
        ens_long.append(sum(w[k] * proba[k][j] for k in models))
        ens_full.append(sum(w[k] * proba_full[k][j] for k in models))
    ens_long = np.array(ens_long)
    ens_full = np.array(ens_full)

    out = pd.DataFrame({"date": dates.iloc[pred_idx].to_numpy()})
    for name in models:
        out[f"p_{name}"] = proba[name]
        out[f"cls_{name}"] = np.argmax(proba_full[name], axis=1)
    out["p_ENS"] = ens_long
    # 原文规则：模型输出类别 1 或类别 2 → 买点（三分类 argmax）
    out["cls_ENS"] = np.argmax(ens_full, axis=1)
    out["dyn_weight_xgb"] = [w["XGB"] for w in dynamic_w]
    out["dyn_weight_lgb"] = [w["LGB"] for w in dynamic_w]
    out["dyn_weight_cat"] = [w["CAT"] for w in dynamic_w]

    importance = (pd.Series(fitted["XGB"].feature_importances_, index=cols)
                  .sort_values(ascending=False).head(20))

    meta = dict(feature_cols=cols, thresholds={**thresholds, "ENS": thr_ens},
                val_f1={**val_f1, "ENS": f1_ens_va}, weights0=weights0,
                n_train=len(Xtr), n_val=len(Xva),
                importance={k.replace("z_", ""): float(v) for k, v in importance.items()})
    return out, meta


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------
def summary_metrics(equity: list[dict], trades: list[dict]) -> dict:
    if not equity:
        return {}
    v0 = equity[0]["value"]
    v1 = equity[-1]["value"]
    rets = []
    for a, b in zip(equity[:-1], equity[1:]):
        if a["value"]:
            rets.append(b["value"] / a["value"] - 1)
    total = v1 / v0 - 1
    ann = (1 + total) ** (252 / max(len(rets), 1)) - 1 if 1 + total > 0 else -1
    sd = np.std(rets, ddof=1) if len(rets) > 1 else 0
    sharpe = float(np.mean(rets) / sd * math.sqrt(252)) if sd > 0 else 0.0
    peak, mdd = v0, 0.0
    for p in equity:
        peak = max(peak, p["value"])
        mdd = min(mdd, p["value"] / peak - 1)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    return dict(total_return_pct=total * 100, annual_return_pct=ann * 100,
                max_drawdown_pct=abs(mdd) * 100, sharpe=sharpe,
                win_rate_pct=wins / len(trades) * 100 if trades else 0.0,
                total_trades=len(trades),
                calmar=(ann * 100 / abs(mdd * 100)) if mdd < 0 else None,
                avg_hold_days=(np.mean([t["holding_bars"] for t in trades]) if trades else 0))


def run_backtest(feat: pd.DataFrame, signal_df: pd.DataFrame,
                 start: str, end: str, hold: int = 5, max_hold: int = 20,
                 stop_loss: float | None = None, take_profit: float | None = None,
                 buy_fee: float = 0.0, sell_fee: float = 0.0, slippage: float = 0.0,
                 initial_cash: float = INITIAL_CASH):
    """信号生成(t 收盘)与执行(t+1 开盘)严格分离；warmup 段不产生任何交易副作用。"""
    df = feat.copy()
    # signal_df: date / sig(0|1)，由 t 日收盘可知的模型输出映射而来
    sig_map = signal_df.set_index("date")["sig"]
    signal = df["date"].map(sig_map).fillna(0).to_numpy().astype(bool)

    cash = initial_cash
    position = 0
    entry_cost = 0.0
    entry_price = 0.0
    entry_date = None
    entry_bar = -1
    deadline = -1
    pending_buy = False
    pending_exit = False
    trades: list[dict] = []
    detail: list[dict] = []
    equity: list[dict] = []
    last_valid_close = float(df["tl_close"].dropna().iloc[0])
    signal_count = 0
    skipped_signals = 0

    ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)
    end_i = int(np.flatnonzero((df["date"] <= ts_end).to_numpy())[-1])

    def close_position(fill: float, date, i: int, reason: str):
        nonlocal cash, position, entry_cost, entry_price, entry_date, entry_bar
        proceeds = position * fill * (1 - sell_fee)
        pnl = proceeds - entry_cost
        rec = dict(entry_date=entry_date.strftime("%Y-%m-%d"),
                   exit_date=date.strftime("%Y-%m-%d"), side="long", size=position,
                   entry_price=round(entry_price, 4), exit_price=round(fill, 4),
                   pnl=round(pnl, 2), pnl_pct=round(pnl / entry_cost * 100, 4),
                   holding_bars=i - entry_bar, symbol=SYMBOL, symbol_name=SYMBOL_NAME)
        trades.append(rec)
        detail.append({**rec, "exit_reason": reason})
        cash += proceeds
        position = 0
        entry_cost = 0.0

    for i in range(end_i + 1):
        row = df.iloc[i]
        date = row["date"]
        in_eval = ts_start <= date <= ts_end

        # warmup gate：训练/历史段只用于计算好的特征，不产生订单/净值/成交记录
        if not in_eval:
            continue

        close = row["tl_close"]
        open_ = row["tl_open"]
        # 停牌 / 数据缺口：保持持仓，用最近有效收盘价估值，不下单
        if pd.isna(close) or pd.isna(open_):
            equity.append({"date": date.strftime("%Y-%m-%d"),
                           "value": round(cash + position * last_valid_close, 2)})
            continue
        last_valid_close = float(close)

        # 1) 执行昨日挂单（次日开盘成交）
        if pending_buy and position == 0:
            fill = float(open_) * (1 + slippage)
            size = int(cash / (fill * (1 + buy_fee)))
            if size > 0:
                entry_cost = size * fill * (1 + buy_fee)
                cash -= entry_cost
                position = size
                entry_price = fill
                entry_date = date
                entry_bar = i
                deadline = i + hold
            else:
                skipped_signals += 1
            pending_buy = False

        if pending_exit and position > 0:
            close_position(float(open_) * (1 - slippage), date, i, "持有到期")
            pending_exit = False

        # 2) 日内止盈止损（日线近似：用 high/low 判定，同日同时触发止损优先）
        if position > 0 and (stop_loss or take_profit):
            low, high = float(row["tl_low"]), float(row["tl_high"])
            sp = entry_price * (1 - stop_loss) if stop_loss else -np.inf
            tp = entry_price * (1 + take_profit) if take_profit else np.inf
            hit_sl = stop_loss is not None and low <= sp
            hit_tp = take_profit is not None and high >= tp
            if hit_sl:  # 止损优先
                fill = min(float(open_), sp) * (1 - slippage) if float(open_) <= sp else sp * (1 - slippage)
                close_position(fill, date, i, "止损")
            elif hit_tp:
                fill = max(float(open_), tp) * (1 - slippage) if float(open_) >= tp else tp * (1 - slippage)
                close_position(fill, date, i, "止盈")

        # 3) 用 t 日已知信息生成 t+1 日的动作
        sig = bool(signal[i])
        if sig:
            signal_count += 1
        if position > 0:
            if sig:  # 持有中再出现买点 → 顺延（原文规则）
                deadline = max(deadline, min(i + 1 + hold, entry_bar + max_hold))
            if i + 1 >= deadline:
                pending_exit = True
        else:
            if sig:
                pending_buy = True

        equity.append({"date": date.strftime("%Y-%m-%d"),
                       "value": round(cash + position * float(close), 2)})

    # 期末强制平仓（按最后一根 bar 的收盘价）
    if position > 0:
        last = df.iloc[end_i]
        last_close = float(last["tl_close"]) if not pd.isna(last["tl_close"]) else last_valid_close
        close_position(last_close * (1 - slippage), last["date"], end_i, "期末强平")
        equity[-1]["value"] = round(cash, 2)

    return dict(equity=equity, trades=trades, detail=detail, metrics=summary_metrics(equity, trades),
                signal_count=signal_count, skipped_signals=skipped_signals, flat=(position == 0))


def benchmark_curve(feat: pd.DataFrame, start: str, end: str, initial_cash=INITIAL_CASH):
    w = feat[(feat["date"] >= start) & (feat["date"] <= end) & feat["tl_close"].notna()]
    base = float(w["tl_close"].iloc[0])
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(initial_cash * c / base, 2)}
            for d, c in zip(w["date"], w["tl_close"])]


def relabel(feat: pd.DataFrame, q1: float, q2: float) -> pd.DataFrame:
    """标签结构参数的敏感性：改三分类的分位切点，其余数据/处理完全不变。"""
    f = feat.copy()
    dy_past = f["g_30年"].diff(5)
    a = dy_past.rolling(756, min_periods=252).quantile(q1)
    b = dy_past.rolling(756, min_periods=252).quantile(q2)
    fwd = f["fwd_dy30_5"]
    lab = np.where(fwd <= a, 2, np.where(fwd <= b, 1, 0))
    f["label"] = pd.Series(lab, index=f.index).astype(float)
    f.loc[fwd.isna() | a.isna(), "label"] = np.nan
    f["label_long"] = f["label"].isin([1.0, 2.0]).astype(int)
    f.loc[f["label"].isna(), "label_long"] = np.nan
    return f


def classification_metrics(feat: pd.DataFrame, signal_df: pd.DataFrame,
                           start: str, end: str):
    m = ((feat["date"] >= start) & (feat["date"] <= end) & feat["label"].notna())
    j = feat.loc[m, ["date", "label_long"]].merge(signal_df, on="date", how="inner")
    y = j["label_long"].astype(int).to_numpy()
    pred = j["sig"].astype(int).to_numpy()
    return dict(precision_pct=precision_score(y, pred, zero_division=0) * 100,
                recall_pct=recall_score(y, pred, zero_division=0) * 100,
                f1_pct=f1_score(y, pred, zero_division=0) * 100,
                n=len(y), signal_rate_pct=pred.mean() * 100,
                base_rate_pct=y.mean() * 100)


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------
def make_charts(result, bench, sens: pd.DataFrame, feat, res_meta, cls, ext_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    eq = pd.DataFrame(result["equity"])
    eq["date"] = pd.to_datetime(eq["date"])
    b = pd.DataFrame(bench); b["date"] = pd.to_datetime(b["date"])

    # 1 净值 vs 基准 + 回撤
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(eq["date"], eq["value"] / INITIAL_CASH, label="赔率择时策略", color="#c62828", lw=1.8)
    ax[0].plot(b["date"], b["value"] / INITIAL_CASH, label="TL 买入持有基准", color="#607d8b", lw=1.4)
    ax[0].set_title("样本外净值曲线（2024-05-15 ~ 2025-05-14）")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    peak = eq["value"].cummax()
    ax[1].fill_between(eq["date"], (eq["value"] / peak - 1) * 100, 0, color="#26a69a", alpha=0.45)
    ax[1].set_title("回撤 (%)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); plt.show(); fig.savefig(HERE / "tl_equity_drawdown.png", dpi=130); plt.close(fig)

    # 2 参数敏感性热力图（持有期 × 概率阈值，夏普）
    piv = sens[sens["group"] == "持有期×阈值"].pivot(index="hold", columns="threshold", values="sharpe")
    if not piv.empty:
        fig, ax = plt.subplots(figsize=(8, 4.2))
        im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(piv.columns)), [f"{c:.2f}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
        ax.set_xlabel("做多概率阈值"); ax.set_ylabel("持有交易日")
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                ax.text(c, r, f"{piv.values[r, c]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_title("参数敏感性：夏普比率"); fig.colorbar(im, ax=ax)
        fig.tight_layout(); plt.show(); fig.savefig(HERE / "tl_sensitivity_heatmap.png", dpi=130); plt.close(fig)

    # 3 交易盈亏分布
    tr = pd.DataFrame(result["trades"])
    if not tr.empty:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].bar(range(len(tr)), tr["pnl_pct"], color=["#c62828" if x > 0 else "#26a69a" for x in tr["pnl_pct"]])
        ax[0].set_title("逐笔收益率 (%)"); ax[0].axhline(0, color="#333", lw=0.8); ax[0].grid(alpha=0.3)
        ax[1].hist(tr["holding_bars"], bins=range(1, max(tr["holding_bars"]) + 3), color="#5c6bc0")
        ax[1].set_title("持仓交易日分布"); ax[1].grid(alpha=0.3)
        fig.tight_layout(); plt.show(); fig.savefig(HERE / "tl_trade_distribution.png", dpi=130); plt.close(fig)

    # 4 因子重要性
    imp = pd.Series(res_meta["importance"]).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp.values, color="#ef6c00")
    ax.set_title("XGBoost 前 15 个特征重要性"); ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); plt.show(); fig.savefig(HERE / "tl_feature_importance.png", dpi=130); plt.close(fig)
    print("[charts] 4 张 PNG 已生成")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    if not FEATURE_CSV.exists() or not PANEL_CSV.exists():
        panel = load_panel()
        feat = build_features(panel)
    else:
        feat = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
        print(f"[load] {FEATURE_CSV.name}: {feat.shape}, "
              f"{feat['date'].min().date()} ~ {feat['date'].max().date()}")

    print("\n=== 1/4 训练集成模型（严格时序隔离） ===")
    probs, meta = train_and_predict(feat)

    # 主信号：原文规则「模型输出类别 1 或类别 2 → 买点」（三分类 argmax）
    sig_ens = pd.DataFrame({"date": probs["date"],
                            "sig": probs["cls_ENS"].isin([1, 2]).astype(int)})
    thr = meta["thresholds"]["ENS"]   # 概率阈值只用于敏感性对照，不作为主信号

    print("\n=== 2/4 主样本外回测 ===")
    base = run_backtest(feat, sig_ens, OOS_START, OOS_END, hold=5)
    bench = benchmark_curve(feat, OOS_START, OOS_END)
    cls = classification_metrics(feat, sig_ens, OOS_START, OOS_END)
    cls_models = {}
    for k in ("XGB", "LGB", "CAT"):
        sg = pd.DataFrame({"date": probs["date"],
                           "sig": probs[f"cls_{k}"].isin([1, 2]).astype(int)})
        cls_models[k] = classification_metrics(feat, sg, OOS_START, OOS_END)
    # 另一种信号映射：验证集选定的概率阈值规则（对照原文的高查全率）
    sig_thr = pd.DataFrame({"date": probs["date"],
                            "sig": (probs["p_ENS"] >= thr).astype(int)})
    cls_prob_rule = classification_metrics(feat, sig_thr, OOS_START, OOS_END)
    print(f"[OOS] 分类: precision={cls['precision_pct']:.2f}%  recall={cls['recall_pct']:.2f}%  "
          f"f1={cls['f1_pct']:.2f}%  信号率={cls['signal_rate_pct']:.1f}%")
    print(f"[OOS] 策略: {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in base['metrics'].items()}, ensure_ascii=False)}")

    export_results(equity_curve=base["equity"], trade_history=base["trades"], prefix=PREFIX,
                   initial_cash=INITIAL_CASH, start=OOS_START, end=OOS_END,
                   market=None, strategy_name="赔率视角30年国债期货择时（XGB/LGB/CAT动态集成）",
                   symbol=SYMBOL, is_flat_at_end=True)
    pd.DataFrame(base["detail"]).to_csv(HERE / "tl_trades_detail.csv", index=False, encoding="utf-8-sig")

    print("\n=== 3/4 扩展样本外 + 参数敏感性 ===")
    ext = run_backtest(feat, sig_ens, EXT_START, EXT_END, hold=5)
    ext_bench = benchmark_curve(feat, EXT_START, EXT_END)
    ext_cls = classification_metrics(feat, sig_ens, EXT_START, EXT_END)
    print(f"[EXT] {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in ext['metrics'].items()}, ensure_ascii=False)}")

    rows = []

    def add(group, label, r, **kw):
        m = r["metrics"]
        rows.append(dict(group=group, param=label,
                         annual_return_pct=round(m.get("annual_return_pct", 0), 3),
                         sharpe=round(m.get("sharpe", 0), 3),
                         max_drawdown_pct=round(m.get("max_drawdown_pct", 0), 3),
                         win_rate_pct=round(m.get("win_rate_pct", 0), 2),
                         total_trades=m.get("total_trades", 0),
                         total_return_pct=round(m.get("total_return_pct", 0), 3),
                         hold=kw.get("hold", 5), threshold=kw.get("threshold", thr)))

    add("基准", "H=5 / argmax 分类信号", base, hold=5, threshold=thr)
    # 信号映射规则：原文逐字口径 vs 验证集选定的概率阈值口径
    r = run_backtest(feat, sig_thr, OOS_START, OOS_END, hold=5)
    add("信号映射", f"概率阈值={thr:.2f}（验证集选定）", r, hold=5, threshold=thr)
    # (a) 持有期 × 概率阈值（概率阈值规则，用于对照主信号）
    for h in (3, 5, 10):
        for t in (0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
            sg = pd.DataFrame({"date": probs["date"], "sig": (probs["p_ENS"] >= t).astype(int)})
            r = run_backtest(feat, sg, OOS_START, OOS_END, hold=h)
            add("持有期×阈值", f"H={h} / thr={t:.2f}", r, hold=h, threshold=t)
    # (b) 模型选择
    for k in ("XGB", "LGB", "CAT"):
        sg = pd.DataFrame({"date": probs["date"],
                           "sig": probs[f"cls_{k}"].isin([1, 2]).astype(int)})
        r = run_backtest(feat, sg, OOS_START, OOS_END, hold=5)
        add("模型", k, r)
    # (c) 止盈止损
    for sl, tp in [(None, None), (0.010, None), (0.015, None), (0.020, None),
                   (None, 0.02), (None, 0.03), (0.015, 0.03), (0.010, 0.02)]:
        r = run_backtest(feat, sig_ens, OOS_START, OOS_END, hold=5,
                         stop_loss=sl, take_profit=tp)
        sl_s = "无" if sl is None else f"{sl * 100:.1f}%"
        tp_s = "无" if tp is None else f"{tp * 100:.1f}%"
        add("止盈止损", f"止损={sl_s} / 止盈={tp_s}", r)
    # (d) 交易成本
    for fee_bps in (0, 2, 5):
        f = fee_bps / 10000
        r = run_backtest(feat, sig_ens, OOS_START, OOS_END, hold=5,
                         buy_fee=f, sell_fee=f, slippage=f)
        add("交易成本", f"双边 {fee_bps}bp + 滑点", r)
    # (e) 持仓续期上限
    for cap in (5, 10, 20, 40):
        r = run_backtest(feat, sig_ens, OOS_START, OOS_END, hold=5, max_hold=cap)
        add("续期上限", f"最长持有 {cap} 日", r)
    # (f) 标签结构：三分类分位切点（需重新训练，检验"信号定义"本身的稳健性）
    for q1, q2 in ((0.15, 0.35), (0.20, 0.40), (0.25, 0.45)):
        fv = relabel(feat, q1, q2)
        pv, mv = train_and_predict(fv, verbose=False)
        sg = pd.DataFrame({"date": pv["date"], "sig": pv["cls_ENS"].isin([1, 2]).astype(int)})
        r = run_backtest(feat, sg, OOS_START, OOS_END, hold=5)
        add("标签分位", f"q20→{q1:.2f} / q40→{q2:.2f}", r)
        if (q1, q2) == (0.20, 0.40):
            print(f"[check] 标签分位复算与原基准一致性: "
                  f"{abs(r['metrics']['annual_return_pct'] - base['metrics']['annual_return_pct']):.4f} pct")

    sens = pd.DataFrame(rows)
    sens.to_csv(HERE / "tl_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"[sensitivity] {len(sens)} 组参数已测试")

    bench_metrics = summary_metrics(bench, [])
    ext_bench_metrics = summary_metrics(ext_bench, [])
    payload = dict(
        config=dict(train_start=TRAIN_START, train_end=TRAIN_END, oos_start=OOS_START,
                    oos_end=OOS_END, ext_start=EXT_START, ext_end=EXT_END,
                    symbol=SYMBOL, initial_cash=INITIAL_CASH, hold=5,
                    threshold=thr, signal_rule="三分类 argmax ∈ {1,2}",
                    max_hold=20, cost_bps=0, forward_days=5),
        model_meta=meta,
        classification=dict(ensemble=cls, models=cls_models,
                            prob_rule=cls_prob_rule, extended=ext_cls),
        strategy=base["metrics"], extended=ext["metrics"],
        benchmark=bench_metrics, extended_benchmark=ext_bench_metrics,
        excess_vs_benchmark_pct=base["metrics"]["total_return_pct"] - bench_metrics["total_return_pct"],
        excess_extended_pct=ext["metrics"]["total_return_pct"] - ext_bench_metrics["total_return_pct"],
        benchmark_curve=bench, extended_equity=ext["equity"],
        extended_trades=ext["trades"],
        signals=dict(count=base["signal_count"], skipped=base["skipped_signals"]),
        article=dict(annual_return_pct=12.58, sharpe=1.54, max_drawdown_pct=6.95,
                     win_rate_pct=67.23, precision_pct=59.42, recall_pct=81.46, f1_pct=68.72),
    )
    (HERE / "tl_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")

    print("\n=== 4/4 生成图表 ===")
    make_charts(base, bench, sens, feat, meta, cls, ext["metrics"])
    print("\n完成。运行 python tl_render_dashboard.py 生成中文仪表盘。")


if __name__ == "__main__":
    main()
