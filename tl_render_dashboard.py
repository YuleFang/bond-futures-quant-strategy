"""
tl_render_dashboard.py — 把回测三件套 + 策略设计 + 参数敏感性渲染为中文 HTML 仪表盘。
运行：python tl_render_dashboard.py
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from render_dashboard import build_dashboard_data, render_dashboard

HERE = Path(__file__).resolve().parent
LANG = "zh"

WIND_ROWS = [
    ("中债国债到期收益率:30年", "S0059752", "w.edb / 日", "预测目标与标签构造；方向与久期"),
    ("中债国债到期收益率:10年 / 1年", "S0059749 / S0059744", "w.edb / 日", "期限利差、曲线斜率、短端锚"),
    ("国债收益率曲线 3M/6M/3Y/5Y/7Y", "曲线族", "w.edb / 日", "曲线水平/斜率/蝶式因子"),
    ("DR007", "M1006337", "w.edb / 日", "银行间资金价格、杠杆成本"),
    ("FR001/FR007/FDR001/FDR007", "回购定盘利率族", "w.edb / 日", "资金面、跨市场资金背离"),
    ("中债中短期票据AAA 3Y/5Y", "信用债曲线族", "w.edb / 日", "信用利差、风险偏好"),
    ("CPI/PPI/PMI/工业增加值/社融/M2", "宏观族", "w.edb / 月", "增长、通胀、广义流动性"),
    ("债券净融资额 / 中债托管量", "债券族", "w.edb / 日月", "债市供求"),
    ("沪深300收盘 / PE(TTM)", "000300.SH", "w.wsd / 日", "股债相对回报率、情绪面"),
    ("地铁客运量/粗钢产量/SCFI/水泥价格", "高频族", "w.edb / 周旬", "高频基本面"),
    ("TL 主力连续 OHLCV / 持仓 / 结算", "TL.CFE", "w.wsd / 日", "交易标的与技术面因子、回测成交价"),
    ("TL 具体月份合约", "TLxxxx.CFE", "w.wsd / 日", "主力连续换月拼接校正"),
]


def fmt(v, digits=2, suffix=""):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    return f"{float(v):.{digits}f}{suffix}"


def wind_table_html():
    rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in WIND_ROWS
    )
    return f"""
    <div class="bt-custom-wind-wrap" style="overflow-x:auto;">
      <table class="bt-custom-wind-table" style="width:100%;border-collapse:collapse;font-size:13px;line-height:1.6;">
        <thead><tr style="background:rgba(127,127,127,.12);">
          <th style="text-align:left;padding:8px;">指标 / 字段</th>
          <th style="text-align:left;padding:8px;">Wind 代码</th>
          <th style="text-align:left;padding:8px;">接口 / 频率</th>
          <th style="text-align:left;padding:8px;">在策略中的用途</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="font-size:12px;opacity:.7;margin-top:8px;">代码经 Wind EDB 指标检索核验（2026-11-11）；当前账号无中债系列提数权限，
      实际回测使用中债估值中心 / 中国货币网 / 中金所行情的公开同源数据，字段口径一一对应。</p>
    </div>
    """


def main():
    summary = json.loads((HERE / "tl_ensemble_summary.json").read_text(encoding="utf-8"))
    res = json.loads((HERE / "tl_results.json").read_text(encoding="utf-8"))
    sens = pd.read_csv(HERE / "tl_sensitivity.csv")
    eq = pd.read_csv(HERE / "tl_ensemble_equity.csv")
    bench = pd.DataFrame(res["benchmark_curve"])

    s, a = summary["summary"], res["article"]
    cls = res["classification"]["ensemble"]
    cfg = res["config"]
    st = res["strategy"]
    ext = res["extended"]

    cmp_rows = [
        ("年化收益", f"{a['annual_return_pct']:.2f}%", fmt(st.get("annual_return_pct"), 2, "%"),
         fmt(st.get("annual_return_pct", 0) - a["annual_return_pct"], 2, "pct")),
        ("夏普比率", f"{a['sharpe']:.2f}", fmt(st.get("sharpe"), 2),
         fmt(st.get("sharpe", 0) - a["sharpe"], 2)),
        ("最大回撤", f"{a['max_drawdown_pct']:.2f}%", fmt(st.get("max_drawdown_pct"), 2, "%"),
         fmt(st.get("max_drawdown_pct", 0) - a["max_drawdown_pct"], 2, "pct")),
        ("交易胜率", f"{a['win_rate_pct']:.2f}%", fmt(st.get("win_rate_pct"), 2, "%"),
         fmt(st.get("win_rate_pct", 0) - a["win_rate_pct"], 2, "pct")),
        ("做多信号 Precision", f"{a['precision_pct']:.2f}%", fmt(cls.get("precision_pct"), 2, "%"),
         fmt(cls.get("precision_pct", 0) - a["precision_pct"], 2, "pct")),
        ("做多信号 Recall", f"{a['recall_pct']:.2f}%", fmt(cls.get("recall_pct"), 2, "%"),
         fmt(cls.get("recall_pct", 0) - a["recall_pct"], 2, "pct")),
        ("做多信号 F1", f"{a['f1_pct']:.2f}%", fmt(cls.get("f1_pct"), 2, "%"),
         fmt(cls.get("f1_pct", 0) - a["f1_pct"], 2, "pct")),
    ]
    body = "\n".join(
        "<tr>" + "".join(f"<td style='padding:6px 8px;'>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in cmp_rows
    )
    verdict = ("方向一致：做多信号具备一定识别力、策略取得正收益" if st.get("annual_return_pct", -99) > 0
               else "方向不一致：本复现未取得正收益")
    pr = res["classification"].get("prob_rule", {})
    art_cmp_html = f"""
    <div class="bt-custom-cmp-wrap" style="overflow-x:auto;">
      <table class="bt-custom-cmp-table" style="width:100%;border-collapse:collapse;font-size:13px;">
        <thead><tr style="background:rgba(127,127,127,.12);">
          <th style="text-align:left;padding:8px;">指标</th>
          <th style="text-align:left;padding:8px;">文章（2024/5/14~2025/5/14）</th>
          <th style="text-align:left;padding:8px;">本次复现</th>
          <th style="text-align:left;padding:8px;">差异</th>
        </tr></thead><tbody>{body}</tbody>
      </table>
      <p style="margin-top:10px;"><strong>判定：{verdict}</strong>；数值不可直接等同，
      差异原因见下方「为何数值不等同于原文」。</p>
      <p><strong>与基准的关系</strong>：同期 TL 买入持有总收益 {fmt(res['benchmark'].get('total_return_pct'), 2, '%')}、
      夏普 {fmt(res['benchmark'].get('sharpe'), 2)}、最大回撤 {fmt(res['benchmark'].get('max_drawdown_pct'), 2, '%')}；
      策略超额收益 {fmt(res.get('excess_vs_benchmark_pct'), 2, 'pct')}。
      即本次复现<strong>绝对收益未跑赢买入持有</strong>，但风险调整后收益（夏普）与回撤控制更好；
      这与原文「各持有期策略均优于基准」的表述不完全一致。</p>
      <p><strong>信号映射口径是最大差异来源</strong>：主规则按原文逐字用三分类 argmax，
      样本外查准 {fmt(cls.get('precision_pct'), 2, '%')} / 查全 {fmt(cls.get('recall_pct'), 2, '%')} /
      F1 {fmt(cls.get('f1_pct'), 2, '%')}；若改用验证集选定的概率阈值
      {fmt(cfg['threshold'], 2)}，则为查准 {fmt(pr.get('precision_pct'), 2, '%')} /
      查全 {fmt(pr.get('recall_pct'), 2, '%')} / F1 {fmt(pr.get('f1_pct'), 2, '%')}
      （信号率 {fmt(pr.get('signal_rate_pct'), 1, '%')}）。
      原文 F1 68.72% 对应的高查全率更接近后者，说明"模型输出→买点"的映射方式必须在实盘前明确定义，
      不能只报告分类指标而不报告信号率。</p>
    </div>
    """

    diff_reasons = f"""
1. **因子集不同**：原文约 100 个因子（含宏观、债券净融资/托管量、地铁客运量、粗钢产量、SCFI、水泥价格、
   30Y 期货价量等，Wind + Akshare 双源，10 年数据）；本次公开数据复现为 {len(res['model_meta']['feature_cols'])} 个
   可核验因子，缺失的另类高频因子未编造。
2. **标签口径**：原文用全样本历史分位数（隐含未来信息）；本实现改为「截至 t 的滚动 756 日分位数」，
   并把 >q40 的上行样本并入类别 0，避免前视偏差。
3. **执行更严格**：原文"依据模型信号持有 5 天"，本实现为 t 日收盘出信号、t+1 日开盘成交、
   持有期内买点顺延（上限 {cfg['max_hold']} 日）、期末强平；并允许叠加止盈止损/成本敏感性。
4. **标的与样本**：TL 于 2023-04-21 上市，训练期（2015~2024）的期货技术面因子在本次以中性值填充，
   与原文"TL 数据较少"的局限一致；原文用固定 XGB，本实现用 XGB/LGB/CAT 动态 F1 加权集成。
5. **基准情形**：0 费用、0 滑点、1 倍价格指数化敞口、不计保证金杠杆，与原文"不考虑交易滑点、
   杠杆率、保证金"一致；加入成本后的结果见「参数敏感性」。
6. **相对基准的差异**：原文称各持有期策略均优于基准；本次复现中 TL 买入持有同期总收益
   {fmt(res['benchmark'].get('total_return_pct'), 2, '%')}，策略 {fmt(st.get('total_return_pct'), 2, '%')}，
   绝对收益略逊，但夏普与最大回撤更优。基准口径（是否有杠杆、是否用具体合约、是否含票息）
   原文未披露，这也是不可直接对照的原因之一。
7. **更长样本外的衰减**：{cfg['ext_start']} ~ {cfg['ext_end']} 策略年化
   {fmt(ext.get('annual_return_pct'), 2, '%')}、基准 {fmt(res['extended_benchmark'].get('total_return_pct'), 2, '%')}（总收益）。
   固定训练截止 2024-05-14 的模型未再重训，市场状态切换后信号失效；生产版应做滚动重训（walk-forward）。
8. **信号映射**：主规则按原文逐字用三分类 argmax（信号率 {fmt(cls.get('signal_rate_pct'), 1, '%')}），
   查全率明显低于原文；改用验证集选定的概率阈值后查全率上升。原文未披露"输出类别"是 argmax
   还是概率阈值，也未披露信号率，这是分类指标最难对齐的一项。
"""

    sens_rows = []
    for _, r in sens.iterrows():
        sens_rows.append({
            "metric": f"{r['group']}｜{r['param']}",
            "values": [
                {"main": fmt(r["annual_return_pct"], 2, "%"), "raw": r["annual_return_pct"]},
                {"main": fmt(r["sharpe"], 2), "raw": r["sharpe"]},
                {"main": fmt(r["max_drawdown_pct"], 2, "%"), "raw": -abs(r["max_drawdown_pct"])},
                {"main": fmt(r["win_rate_pct"], 1, "%"), "raw": r["win_rate_pct"]},
                {"main": str(int(r["total_trades"]))},
            ],
        })

    # 策略 vs 基准对比曲线（归一化到 1）
    base_val = eq["value"].iloc[0]
    strat_pts = [{"date": d, "value": round(v / base_val, 5)} for d, v in zip(eq["date"], eq["value"])]
    bench_base = bench["value"].iloc[0]
    bench_pts = [{"date": str(d)[:10], "value": round(v / bench_base, 5)}
                 for d, v in zip(bench["date"], bench["value"])]

    design_text = f"""
**预测目标**：未来 {cfg['forward_days']} 个交易日 30Y 国债收益率变动方向，三分类（0 震荡/非买点、1 小幅下行、2 大幅下行）。

**模型**：XGBoost / LightGBM / CatBoost；按自然年分块，每年前 80% 训练、后 20% 验证；
训练集对类别 1/2 上采样；验证集 F1 决定模型权重；样本外按滚动 120 日（滞后 5 日已实现标签）动态更新权重。
训练集 {res['model_meta']['n_train']} 条，验证集 {res['model_meta']['n_val']} 条。

**信号（主规则，原文口径）**：三分类模型输出 argmax 为类别 1 或 2 → 买点（只做多，赔率视角，不识别做空）。
另设概率阈值规则 p_long = P(1)+P(2) ≥ {fmt(cfg['threshold'], 2)} 仅用于敏感性对照；
该阈值在验证集上以「最大化做多 F1 且信号率 ≤ 60%」选出，防止 F1 最大化退化为几乎全做多。

**交易规则**：
- 开仓：t 日收盘确认买点 → t+1 日开盘买入 TL 主力连续。
- 平仓：持有满 {cfg['hold']} 个交易日 → 次日开盘卖出；持有期内再次出现买点则顺延（最长 {cfg['max_hold']} 日）。
- 止盈止损：基准关闭；增强版支持以当日最低价触发止损、当日最高价触发止盈，同日同时触发止损优先，跳空按开盘价成交。
- 仓位：单标的满仓、1 倍价格指数化敞口，不计保证金与合约乘数；基准 0 费用 0 滑点。
- 期末：最后一根 K 线收盘强制平仓。

**区间**：训练 {cfg['train_start']} ~ {cfg['train_end']}；样本外主区间 {cfg['oos_start']} ~ {cfg['oos_end']}；
扩展样本外 {cfg['ext_start']} ~ {cfg['ext_end']}。标的 {cfg['symbol']}（30 年期国债期货主力连续）。
"""

    limitations = """
- **前视偏差已尽力消除**：所有特征只用 t 及以前数据；标签阈值用滚动历史分位；信号与执行相隔一根 K 线；动态权重只用已实现标签。
- **日线近似**：止盈止损用最高/最低价判定，日内先后顺序不可知；未建模盘口冲击、保证金追缴、换月移仓价差。
- **主力连续拼接**：TL0 连续合约在换月处存在跳空，可能高估或低估 5 日持仓收益；生产环境应改用具体月份合约并按持仓量换月。
- **生存/样本偏差**：TL 2023-04-21 才上市，训练期几乎全为利率下行样本，模型天然偏向做多；这与原文"不善于识别做空场景"的观察一致。
- **过拟合风险**：89 个特征 / 约 2200 个训练样本，且未做原文的完整 KS 分布稳定性筛选与贝叶斯调参；样本外可能显著弱于样本内。
- **绩效口径**：夏普按 252 日、无风险利率 0 计算；年化由区间净值复利折算，短区间年化对小样本极敏感。
"""

    extra = [
        {"type": "text", "tab": "overview", "title": "结论速览",
         "text": f"样本外（{cfg['oos_start']} ~ {cfg['oos_end']}）年化 {fmt(st.get('annual_return_pct'), 2, '%')}、"
                 f"夏普 {fmt(st.get('sharpe'), 2)}、最大回撤 {fmt(st.get('max_drawdown_pct'), 2, '%')}、"
                 f"胜率 {fmt(st.get('win_rate_pct'), 1, '%')}，共 {st.get('total_trades', 0)} 笔完整交易；"
                 f"同期 TL 买入持有基准总收益 {fmt(res['benchmark'].get('total_return_pct'), 2, '%')}，"
                 f"策略超额 {fmt(res.get('excess_vs_benchmark_pct'), 2, 'pct')}。"
                 f"扩展样本外（{cfg['ext_start']} 起）年化 {fmt(ext.get('annual_return_pct'), 2, '%')}、"
                 f"夏普 {fmt(ext.get('sharpe'), 2)}，同期基准总收益 "
                 f"{fmt(res['extended_benchmark'].get('total_return_pct'), 2, '%')}，"
                 f"策略超额 {fmt(res.get('excess_extended_pct'), 2, 'pct')}——该区间市场整体回落，"
                 f"策略绝对收益转负但相对基准仍略优，说明固定模型在更长样本外存在明显衰减。"},
        {"type": "line_chart", "tab": "overview", "title": "策略净值 vs TL 买入持有（归一化）",
         "subtitle": "样本外主区间，起点 = 1.00",
         "series": [{"name": "赔率择时策略", "points": strat_pts},
                    {"name": "TL 买入持有", "points": bench_pts}]},
        {"type": "text", "tab": "design", "title": "核心思路、信号逻辑与交易规则", "text": design_text},
        {"type": "custom_html", "tab": "design", "title": "需要从 Wind 获取的数据字段及用途",
         "width": "full", "html": wind_table_html()},
        {"type": "text", "tab": "design", "title": "绩效评估指标口径",
         "text": "累计收益、年化收益（252 日复利折算）、夏普比率（日均超额/日波动×√252，无风险利率 0）、"
                 "最大回撤（净值相对历史峰值）、Calmar（年化/最大回撤）、交易胜率（净收益>0 的平仓笔数占比）、"
                 "交易次数、平均持仓交易日；分类侧报告做多信号的 Precision / Recall / F1。"
                 "另以 TL 买入持有作为基准，并检验参数敏感性与交易成本敏感性。"},
        {"type": "metric_table", "tab": "sensitivity", "title": "参数敏感性全表",
         "subtitle": "持有期×阈值 / 单模型 / 止盈止损 / 交易成本 / 续期上限",
         "columns": ["参数组｜设置", "年化收益", "夏普", "最大回撤", "胜率", "交易数"],
         "rows": sens_rows},
        {"type": "text", "tab": "sensitivity", "title": "敏感性处理结论",
         "text": f"共重跑 {len(sens)} 组参数。处理方法：① 主信号采用原文的 argmax 分类，"
                 f"概率阈值 {fmt(cfg['threshold'], 2)} 只在验证集用「F1 最大化 + 信号率上限」选出，"
                 "不用样本外调参；"
                 "② 对持有期 3/5/10 日、阈值 0.30~0.60、信号映射口径、三类单模型、8 组止盈止损、"
                 "0/2/5bp 成本、续期上限 5~40 日，以及三分类标签分位切点（重新训练）逐一重跑，"
                 "观察绩效是否只在单一参数点上成立；"
                 "③ 参数平台（相邻参数绩效接近、符号不翻转）视为稳健，孤立尖峰视为过拟合信号；"
                 "④ 额外用 2025/5 之后的扩展样本外做时间外检验。"
                 "从全表看：持有期与阈值构成较平缓的绩效平台（多数年化为正），但"
                 "信号映射口径与止盈止损对收益影响最大；胜率对小样本极其敏感（部分组合 100%），"
                 "且加入 5bp 成本后绩效明显下降。<strong>注意：全表中在样本外扫描出来的最优点"
                 "仅用于诊断参数敏感性，属于数据窥探，不可作为可交易的选参依据</strong>；"
                 "实盘参数只能由验证集或滚动样本内决定。"},
        {"type": "custom_html", "tab": "conclusion", "title": "与文章结论的对照",
         "width": "full", "html": art_cmp_html},
        {"type": "text", "tab": "conclusion", "title": "为何数值不等同于原文", "text": diff_reasons},
        {"type": "text", "tab": "conclusion", "title": "已知偏差与局限", "text": limitations},
        {"type": "text", "tab": "conclusion", "title": "免责声明",
         "text": "以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。"
                 "任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。"
                 "过往表现不预示未来收益。"},
    ]

    ui = {"tabs": [
        {"id": "overview", "label": "回测总览"},
        {"id": "design", "label": "策略设计与 Wind 字段"},
        {"id": "sensitivity", "label": "参数敏感性"},
        {"id": "conclusion", "label": "结论对照与局限"},
    ], "active_tab": "overview", "subtitle": "30 年期国债期货 · 赔率视角机器学习择时"}

    data = build_dashboard_data(
        equity_csv=str(HERE / "tl_ensemble_equity.csv"),
        trades_csv=str(HERE / "tl_ensemble_trades.csv"),
        summary_json=str(HERE / "tl_ensemble_summary.json"),
        language=LANG,
        market=None,
        extra_modules=extra,
        ui_overrides=ui,
    )
    out = render_dashboard(data, output_path=str(HERE / "index.html"),
                           template_path=str(HERE / "dashboard_template.html"))
    print(f"[dashboard] {out}  size={out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
