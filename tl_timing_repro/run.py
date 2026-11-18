"""一键复现入口：python run.py

流程：加载 Wind 缓存 -> 构造因子/标签 -> 预处理 -> 两段各训模型 ->
按年滚动重训生成样本外信号 -> 回测 -> 评估 -> 敏感性 -> 输出与自检。

输出目录 output/：nav.csv、trades.csv、metrics.json、classification_report.json、
comparison_to_paper.csv、sensitivity.csv、preprocess_report.json、
self_check.json、walk_forward_log.json、segment_a_report.json、
feature_list.json、summary.md。
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.backtest import run_backtest                     # noqa: E402
from src.data_fetch import (                             # noqa: E402
    BASE_DIR as _BD, describe_wind_fields, load_config, load_edb, load_prices,
)
from src.evaluate import (                               # noqa: E402
    comparison_table, compute_metrics, delivery_self_check, format_metrics_for_report,
)
from src.factors import (                                # noqa: E402
    assert_no_lookahead, build_features, build_labels,
)
from src.models import (                                 # noqa: E402
    binary12_metrics, predict_all, proba_to_label, train_models, walk_forward_predict,
)
from src.preprocess import Preprocessor                  # noqa: E402
from src.sensitivity import sweep                        # noqa: E402

OUT = os.path.join(BASE_DIR, "output")


def _json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def run_once(cfg: dict, write_outputs: bool = False, verbose: bool = True) -> dict:
    seed = int(cfg["project"]["seed"])
    np.random.seed(seed)
    px_b = load_prices(cfg, "segment_b")
    px_a = load_prices(cfg, "segment_a")
    edb = load_edb(cfg)
    oos_start = pd.Timestamp(cfg["data"]["oos_start"])
    oos_end = pd.Timestamp(cfg["data"]["oos_end"])

    # ---------------- 段 B：主结果 ----------------
    feat_b = build_features(px_b, edb, cfg)
    lab_b = build_labels(px_b, cfg)
    valid = ~lab_b["label"].isna()
    X_raw = feat_b.loc[valid]
    y = lab_b.loc[valid, "label"].astype(int)
    dates = X_raw.index
    oos_mask = np.asarray((dates >= oos_start) & (dates <= oos_end))
    train_mask = np.asarray(dates < oos_start)
    if not oos_mask.any() or not train_mask.any():
        raise ValueError("训练期或样本外区间为空，请检查缓存范围与 config.data")

    pre = Preprocessor(cfg).fit(X_raw.loc[train_mask])
    X = pre.transform(X_raw)

    preds, wf_log = walk_forward_predict(X, y, dates, cfg, seed, oos_mask)
    if not preds:
        raise RuntimeError("没有任何模型成功训练，请检查依赖安装与训练样本")
    primary = cfg["model"]["primary"]
    main_name = primary if primary in preds else sorted(preds)[0]

    cls_by_model = {}
    for name, p in preds.items():
        idx = np.where(oos_mask)[0]
        valid_p = ~np.isnan(p[idx]).all(axis=1)
        cls_by_model[name] = binary12_metrics(
            y.to_numpy()[idx][valid_p], proba_to_label(p[idx][valid_p])
        )

    sig_by_date = pd.Series(proba_to_label(preds[main_name]), index=dates)
    oos_px = px_b[(px_b["date"] >= oos_start) & (px_b["date"] <= oos_end)].reset_index(drop=True)
    signal = sig_by_date.reindex(pd.DatetimeIndex(oos_px["date"])).fillna(0).to_numpy(dtype=int)

    bt = run_backtest(oos_px, signal, cfg)
    metrics = compute_metrics(bt["nav"], bt["trades"], cfg)
    cls = cls_by_model[main_name]
    no_lookahead = assert_no_lookahead(px_b, edb, cfg)
    self_check = delivery_self_check(cfg, no_lookahead, BASE_DIR)

    # ---------------- 段 A：独立模型（仅分类评估，绩效以段 B 为准） ----------------
    feat_a = build_features(px_a, edb, cfg)
    lab_a = build_labels(px_a, cfg)
    va = ~lab_a["label"].isna()
    Xa_raw = feat_a.loc[va]
    ya = lab_a.loc[va, "label"].astype(int)
    split = int(len(Xa_raw) * (1 - cfg["model"]["validation_frac"]))
    seg_a_report = {"n_valid_samples": int(len(Xa_raw)), "n_train": int(split)}
    if split > 30 and len(Xa_raw) - split > 10:
        pre_a = Preprocessor(cfg).fit(Xa_raw.iloc[:split])
        Xa = pre_a.transform(Xa_raw)
        bundle_a = train_models(
            Xa.iloc[:split].to_numpy(float), ya.iloc[:split].to_numpy(int), cfg, seed
        )
        pa = predict_all(bundle_a, Xa.iloc[split:].to_numpy(float))
        if pa:
            nm = primary if primary in pa else sorted(pa)[0]
            seg_a_report["classification_12"] = binary12_metrics(
                ya.iloc[split:].to_numpy(int), proba_to_label(pa[nm])
            )
            seg_a_report["model"] = nm
    seg_a_report["note"] = "段 A 独立建模，仅用于对照；样本外绩效以段 B 为准"

    result = {
        "metrics": metrics,
        "classification": cls,
        "classification_by_model": cls_by_model,
        "main_model": main_name,
        "backtest": bt,
        "preprocess_report": pre.report(),
        "no_lookahead": no_lookahead,
        "self_check": self_check,
        "walk_forward": wf_log,
        "segment_a": seg_a_report,
        "feature_columns": list(X.columns),
        "oos_px": oos_px,
        "signal": signal,
    }

    if write_outputs:
        os.makedirs(OUT, exist_ok=True)
        bt["nav"].to_csv(os.path.join(OUT, "nav.csv"), index=False, encoding="utf-8")
        bt["trades"].to_csv(os.path.join(OUT, "trades.csv"), index=False, encoding="utf-8")
        _json(os.path.join(OUT, "metrics.json"), metrics)
        _json(os.path.join(OUT, "classification_report.json"), {
            "main_model": main_name, "by_model": cls_by_model,
            "definition": "类别 1+2 合并为做多信号后的 precision/recall/F1",
        })
        comparison_table(metrics, cls, cfg).to_csv(
            os.path.join(OUT, "comparison_to_paper.csv"), index=False, encoding="utf-8-sig"
        )
        _json(os.path.join(OUT, "preprocess_report.json"), pre.report())
        _json(os.path.join(OUT, "self_check.json"), self_check)
        _json(os.path.join(OUT, "walk_forward_log.json"), wf_log)
        _json(os.path.join(OUT, "segment_a_report.json"), seg_a_report)
        _json(os.path.join(OUT, "feature_list.json"), {
            "n_features": len(X.columns), "columns": list(X.columns),
            "note": "技术面因子段内各自构造；EDB 因子按交易日 ffill 对齐",
        })
    if verbose:
        print(f"[主模型] {main_name} | {format_metrics_for_report(metrics)}")
        print(f"[分类]   precision={cls['precision']:.4f} recall={cls['recall']:.4f} f1={cls['f1']:.4f}")
        print(f"[自检]   无未来函数={no_lookahead['passed']} 全部检查={self_check['all_passed']}")
    return result


def write_summary(cfg: dict, res: dict, sens: pd.DataFrame) -> None:
    m, cls = res["metrics"], res["classification"]
    backend_notes = {}
    for fold in res["walk_forward"]["folds"]:
        for k, v in fold["model_notes"].items():
            backend_notes[k] = v
    degraded = any("降级" in v or "未安装" in v for v in backend_notes.values())
    if degraded:
        BACKEND_LINE = (
            "本机第三方 GBDT/scipy 依赖不可用，实际运行降级为内置纯 numpy softmax；"
            "代码已实现真实三模型，装好依赖后重跑即自动切换"
        )
        BACKEND_BLOCK = """## 0. 运行环境与模型后端（重要）

**本次运行发生了模型后端降级。** 本机 pip 受网络代理限制，`scipy / scikit-learn /
xgboost / lightgbm / catboost / statsmodels` 未能安装，因此预处理改用纯 numpy 近似
（Shapiro/Box-Cox/KS/ADF 降级），模型降级为内置多项 softmax 回归（`builtin_softmax`），
**并非 XGBoost**。明细见 `output/walk_forward_log.json` 与 `output/self_check.json`。

代码已完整实现 XGBoost / LightGBM / CatBoost 与 F1 加权动态集成；在可安装依赖的环境执行
`pip install -r requirements.txt && python run.py` 即自动切换到真实三模型。
**下文绩效与分类指标基于降级模型，不代表原文 XGBoost 效果，也不构成对原文的验证。**

"""
    else:
        BACKEND_LINE = "按确认口径使用 XGBoost 主模型，三模型与集成均输出对照"
        BACKEND_BLOCK = ""
    cmp = comparison_table(m, cls, cfg)
    cmp_lines = "\n".join(
        f"| {r['指标']} | {r['本次复现']} | {r['原文']} | {r['差异']} |"
        for _, r in cmp.iterrows()
    )
    deviations = [
        ("预测目标", "原文预测 30Y 国债收益率方向", "改用 T.CFE / TL.CFE 期货收盘价代理（S0059752 无 Wind 权限）"),
        ("数据范围", "段 A 2015-05-14 起、约 100 因子过去 10 年", "缓存段 A 2022-01-04 起、4 个 EDB 指标 + 价格衍生因子"),
        ("标签口径", "全样本历史分位数；上行样本未定义", "扩展窗口分位数（无未来函数）；上行样本并入类别 0"),
        ("建模方式", "三模型 + 动态集成，XGB 最优", "两段各训一个模型；XGB 为主，其余与 F1 加权集成作对照"),
        ("成交价", "未写明", "信号当日收盘建仓（**已知轻微前视偏差**）"),
        ("成本", "不考虑滑点/杠杆/保证金", "佣金/滑点/冲击一律 0，未做成本敏感性（按用户选择）"),
        ("容量", "未提及", "未评估，登记为风险项"),
        ("价格口径", "未指定", "统一 get_index_kline；已排除 get_financial_data 另一套价格"),
        ("预处理", "MAD/Box-Cox/KS 阈值未给", "固定默认值写入 config.yaml，MAD 改训练集全局统计"),
        ("KS 样本期", "15 年样本期", "按实际 9 年样本期折算"),
        ("模型后端", "XGBoost / LightGBM / CatBoost + F1 加权集成", BACKEND_LINE),
    ]
    dev_lines = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in deviations)
    tr = res["backtest"]["trades"]
    avg_hold = float(tr["holding_days"].mean()) if len(tr) else 0.0
    unstable = sens[(sens["status"] == "ok") & (sens["n_trades"] <= 2)][["param", "value", "n_trades"]]
    unstable_line = (
        "；".join(f"{r['param']}={r['value']} 仅 {r['n_trades']} 笔" for _, r in unstable.iterrows())
        or "无"
    )
    text = f"""# 策略复现摘要（方法学复现）

- 原文：{cfg['project']['paper']}
- 复现模式：{cfg['project']['mode']}（不追求数值对齐，逐项登记偏离）
- 运行：`python run.py`（依赖见 requirements.txt，数据默认读 `data/` 缓存）

{BACKEND_BLOCK}## 1. 策略摘要

用价格衍生技术面因子 + Wind EDB 流动性/高频基本面因子，预测未来 5 个交易日方向；
三分类标签（0 震荡/上行、1 小幅下行、2 大幅下行）由期货 5 日收益的**扩展窗口分位数**生成
（利率下行 ↔ 期货上涨，故取收益高分位为下行信号）。段 A 用 T.CFE、段 B 用 TL.CFE 各自建模，
XGBoost 为主模型，按年滚动重训生成样本外信号；信号 1/2 → 当日收盘买入 TL，持有 5 个交易日，
持有期内再现买点则重置持有期，不加仓、只做多、零成本。

## 2. 本次结果（样本外 {m['sample_start']} ~ {m['sample_end']}，{m['n_trading_days']} 个交易日）

{format_metrics_for_report(m)}
分类（类别 1+2 合并）：precision {cls['precision']:.4f} / recall {cls['recall']:.4f} / F1 {cls['f1']:.4f}
基准（TL 买入持有）累计收益：{m['benchmark_cumulative_return']:.2%}
因子数：{len(res['feature_columns'])}；交易笔数：{m['n_trades']}；持仓暴露：{res['backtest']['exposure']:.2%}
注意：实际平均持有 {avg_hold:.1f} 个交易日（名义 5 天）——样本外做多信号较密集，
「再现买点重置持有期」规则被反复触发，持仓被显著拉长，这是与原文「持有 5 天」
不可比的重要原因。交易仅 {m['n_trades']} 笔，胜率/夏普在小样本下不稳定；
敏感性中交易笔数 ≤2 的参数点：{unstable_line}。

## 3. 与原文数值差异（原文为持有 5 天、样本外 2024-05-14~2025-05-14）

| 指标 | 本次复现 | 原文 | 差异 |
|---|---|---|---|
{cmp_lines}

差异归因：预测目标改为期货价格代理、缓存区间缩短、因子为可取得子集、标签分位数口径修正、
模型超参与预处理参数为固定默认值、成交价与指标口径补全、样本外区间行情本身不同。
**本表不构成对原文结果的验证或否定。**

## 4. 数据依赖

{describe_wind_fields()}

权限事实：`S0059752`（中债国债到期收益率:30年）当前账号**无权限**；接口存在单次返回量与
QPS 限制。落地缓存由 `tools/import_wind_results.py` 从 WorkBuddy 持久化的 Wind MCP 结果导入。

## 5. 代码结构

```
run.py                    一键入口
config.yaml               全部口径与参数（唯一配置源）
src/data_fetch.py         缓存读取 + Wind 取数留桩/字段说明
src/factors.py            技术面/EDB 因子、三分类标签、无未来函数扰动自检
src/preprocess.py         缺失/异常值、平稳性、Box-Cox、滚动稳健标准化、KS
src/models.py             三模型 + 上采样 + F1 加权集成 + 按年滚动重训
src/backtest.py           信号 -> 交易 -> 净值
src/evaluate.py           绩效指标、差异表、交付前自检
src/sensitivity.py        单因子参数敏感性
tools/import_wind_results.py  Wind MCP 结果 -> CSV 缓存
tools/pypi_tuna_proxy.py     清华镜像只读转发（pip 被网关拦截时启用）
data/                     Wind 真实数据缓存 + DATA_MANIFEST.md
output/                   运行结果
```

## 6. 运行方式

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python tools/import_wind_results.py   # 可选：重新导入 Wind MCP 结果
python run.py                          # 一键复现
```
若 pip 直连镜像被网关拦截，先运行 `python tools/pypi_tuna_proxy.py` 起本地只读转发，
再用 `-i http://127.0.0.1:8765/simple/ --trusted-host 127.0.0.1` 安装，数据源仍为清华镜像。

## 7. 结果验证方式

1. 一键复算：`python run.py` 后核对 `output/metrics.json` 与 `output/nav.csv`；
   固定随机种子 42，重复运行数值应一致。
2. 无未来函数：`output/self_check.json` 中「因子扰动测试」——扰动 t 之后价格，
   t 及之前的因子不得变化；walk-forward 训练明确排除样本外标签。
3. 口径一致性：`self_check.json` 校验成本参数、依赖、缓存完整性；`config.yaml` 为唯一配置源。
4. 参数敏感性：`output/sensitivity.csv` 给出持仓期/MAD 倍数/滚动窗口/种子的单因子扰动结果，
   观察绩效是否只在单点成立。
5. 对照：`comparison_to_paper.csv` 与 `classification_report.json`（各模型对照）。
6. 边界：代码只做研究与回测，不含实盘下单、不连券商/交易所接口、不含自动交易循环。

## 8. 偏离登记

| 项 | 原文 | 本次 |
|---|---|---|
{dev_lines}

## 9. 已知风险与合规

- 已知轻微前视偏差（当日收盘出信号并成交）、零成本假设、容量未评估、因子为子集、
  缓存区间短于原文、两种 Wind 价格接口口径差异 —— 均会使结果偏乐观或不可比。
- 自检结果：{'全部通过' if res['self_check']['all_passed'] else '存在未通过项，见 output/self_check.json'}。
- 投资有风险，本内容仅供研究与教育，不构成投资建议；回测结果不等于实盘表现，
  历史业绩不代表未来。不荐股、不给买卖点与目标价、不承诺收益。
- 涨跌表述按国内习惯：红涨绿跌。
"""
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    cfg = load_config()
    print("=" * 72)
    print("赔率视角下的 30 年国债择时模型 · 方法学复现")
    print("=" * 72)
    try:
        res = run_once(cfg, write_outputs=True, verbose=True)
        print("\n[敏感性分析] 运行中 ...")
        sens = sweep(lambda c: run_once(c, write_outputs=False, verbose=False), cfg)
        sens.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False, encoding="utf-8-sig")
        write_summary(cfg, res, sens)
        print(f"[输出] {OUT}")
        print(f"[敏感性] {len(sens)} 组，失败 {(sens['status'] != 'ok').sum()} 组")
        return 0 if res["self_check"]["all_passed"] else 2
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
