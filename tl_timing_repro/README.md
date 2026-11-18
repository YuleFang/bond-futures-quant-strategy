# 赔率视角下的 30 年国债择时模型 · 方法学复现

原文：浙商固收《赔率视角下的30年国债择时模型》（覃汉 / 章恒豪，2025-06-20）
<https://mp.weixin.qq.com/s/AzuocHvvEgQ24616w20AGA>

本仓库是**方法学复现**：完整实现「数据处理 → 因子 → 三分类标签 → 集成模型 →
择时信号 → 回测 → 评价」链路，参数固定并逐项登记与原文的偏离；**不追求数值对齐**。
所有口径以 `config.yaml` 为唯一配置源，来源为 2026-11-17 与用户逐题确认的 7 组决策。

> 代码只做研究与回测，不含实盘下单、不连券商/交易所接口、不含自动交易循环。

---

## 1. 策略摘要

| 项 | 本次实现 |
|---|---|
| 预测目标 | 未来 5 个交易日方向；用 `T.CFE`（段 A）/ `TL.CFE`（段 B）**期货收盘价代理**利率方向（30Y 收益率 Wind 指标无权限） |
| 标签 | `r_t = close[t+5]/close[t] − 1`；**扩展窗口**分位数（只用 t 之前已完全实现的样本）。利率下行 ↔ 期货上涨，故 `r` 分位 >0.80 → 类别 2 大幅下行，0.60~0.80 → 类别 1 小幅下行，<0.60 → 类别 0（震荡 + 上行样本并入 0） |
| 因子 | 价格衍生技术面因子（收益/波动/均线偏离/RSI/振幅/量能/偏度等）+ Wind EDB 的 DR007、R001、水泥价格指数、SCFI；全部为可显式写出的公式，不造未定义交叉项；技术面因子段内各自构造、不跨段拼接 |
| 分段 | 段 A `T.CFE` 2015-05-14~2023-04-20（缓存 2022-01-04 起）；段 B `TL.CFE` 2023-04-21~2025-05-14；两段各训一个模型，样本外绩效以段 B 为准 |
| 预处理 | 缺失率>15% 弃列、MAD(n=3) 去极值、Shapiro-Wilk(α=0.05) 决定均值/中位数/KNN(K=5) 填补、Box-Cox 自动估 λ、ADF/KPSS 后差分、滚动稳健标准化 `(X−Median)/IQR`（244 交易日）、KS(α=0.05) 稳定性筛选 |
| 模型 | XGBoost（主）/ LightGBM / CatBoost，库默认超参 + 种子 42 + 验证集 20% early stopping；训练集对类别 1/2 上采样；F1 加权动态集成作对照；不做贝叶斯搜索；按年滚动重训 walk-forward |
| 交易 | 信号 1/2 → **当日收盘**买入 TL；持有 5 个交易日后收盘平仓；持有期内再现买点则重置持有期，不加仓、不重复建仓；100% 名义仓位、只做多、不加杠杆、不计保证金 |
| 成本 | 佣金/滑点/冲击一律 0（按确认口径，不做成本敏感性） |
| 评价 | 年化收益、夏普(rf=0, 252)、最大回撤(净值)、胜率(平仓交易笔数)、累计收益；分类 Precision/Recall/F1（类别 1+2 合并）；基准 = 样本外 TL 买入持有 |
| 样本外 | 2024-05-14 ~ 2025-05-14 |

---

## 2. 数据依赖（只用万得）

| 用途 | Wind 工具 / 代码 | 字段 | 频率 | 目标范围 | 缓存实际范围 | 权限状态 |
|---|---|---|---|---|---|---|
| 段 A 价量 | `get_index_kline` `T.CFE` | TIME/OPEN/MATCH(收)/HIGH/LOW/VOLUME | 日 K | 2015-05-14~2023-04-20 | 2022-01-04~2023-04-20 (314) | 可取（T 上市 2015-03-20） |
| 段 B 价量 | `get_index_kline` `TL.CFE` | 同上 | 日 K | 2023-04-21~2025-05-14 | 2023-04-21~2025-05-14 (497) | 可取（TL 上市 2023-04-21） |
| 流动性 | EDB `M1006337` | DR007 | 日 | 同上 | 837 观测 | 可提数 |
| 流动性 | EDB `M0041652` | R001 银行间质押式回购加权利率:1天 | 日 | 同上 | 837 观测 | 可提数 |
| 高频基本面 | EDB `S5914515` | 中国:水泥价格指数 | 日 | 同上 | 827 观测 | 可提数 |
| 高频基本面 | EDB `S0114089` | SCFI:综合指数 | 周 | 同上 | 167 观测 | 可提数 |
| 高频基本面 | EDB `S5708246` | 粗钢日均产量:重点企业 | 日 | 同上 | 未纳入缓存 | 已验证可取 |
| 标签(原文) | EDB `S0059752` | 中债国债到期收益率:30年 | 日 | — | — | **当前账号无权限**（硬阻塞，已改用期货代理） |
| 结构因子 | EDB `S0059749` 等 | 10Y/1Y 收益率、期限利差 | 日 | — | — | 权限未验证 |

**未纳入本次运行**：宏观经济（PMI/CPI 等）、货币政策与广义流动性、债市供求（每日净融资/托管额）、
情绪面与相对价值（信用利差、股债相对回报、杠杆率）、地铁客运量、粗钢产量、期限利差。
原因：`S0059752` 无权限；其余指标缺完整清单与代码，且 Wind 单次返回量与 QPS 受限。
`src/data_fetch.py` 的 `WIND_FIELD_SPEC` 给出补全字段清单，`--source windpy` / Wind MCP 导出通道
可在有完整权限的环境补齐后重跑。

**接口事实（实测）**

- `get_index_kline` 单次可取数百行且无 100 行截断；`get_financial_data` 单次约 100 行截断。
- 并行调用会触发 QPS 超限，取数须串行 + 分段 + 本地缓存。
- **同一标的两种接口价格口径不同**：2024-05-14 的 `TL.CFE` 收盘，`get_index_kline` 为 105.62，
  `get_financial_data` 为 106.56。本仓库统一采用 `get_index_kline`，并在导入脚本中显式排除
  另一套结果，避免混用。
- 缓存由 `tools/import_wind_results.py` 从 WorkBuddy 持久化的 Wind MCP 结果自动导入，
  来源文件与行数见 `data/DATA_MANIFEST.md`。**不含任何合成或伪造行情数据**。

---

## 3. 代码结构

```
run.py                        一键入口（全流程 + 输出 + 自检）
config.yaml                   全部口径与参数（唯一配置源）
requirements.txt              依赖
src/data_fetch.py             缓存读取、交易日对齐、Wind 取数留桩与字段说明
src/factors.py                技术面/EDB 因子、三分类标签、无未来函数扰动自检
src/preprocess.py             缺失/异常值、平稳性、Box-Cox、滚动标准化、KS
src/models.py                 三模型 + 上采样 + F1 加权集成 + 按年滚动重训
src/backtest.py               信号 → 交易 → 净值/基准
src/evaluate.py               绩效指标、原文差异表、交付前自检
src/sensitivity.py            单因子参数敏感性
tools/import_wind_results.py  Wind MCP 结果 → CSV 缓存（可重跑扩充数据）
tools/pypi_tuna_proxy.py     清华镜像只读转发（pip 被网关拦截时启用）
data/                         Wind 真实数据缓存 + DATA_MANIFEST.md
output/                       运行结果（nav/metrics/分类报告/差异表/敏感性/自检/摘要）
```

---

## 4. 运行方式

```bash
cd C:/Users/Administrator/WorkBuddy/实验室/tl_timing_repro
python -m pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn
python tools/import_wind_results.py     # 可选：从 Wind MCP 结果重新导入缓存
python run.py                           # 一键复现
```

默认读取 `data/` 下的 Wind 缓存（离线可运行）。`config.yaml` 中任何口径改动都会
同步反映到输出。若要补全 2015-05-14 起的全量数据，用 `windpy` 后端或 Wind MCP
导出同名 CSV 覆盖 `data/` 后重跑即可。

若 pip 直连镜像被网关拦截（症状：curl 访问镜像为 200，但 pip 报
`No matching distribution found` / 403），先起随附的只读转发服务，数据源仍是清华镜像：

```bash
python tools/pypi_tuna_proxy.py         # 另一个终端，监听 127.0.0.1:8765
python -m pip install -r requirements.txt \
  -i http://127.0.0.1:8765/simple/ --trusted-host 127.0.0.1
```

**依赖与模型后端（2026-11-17 已完整安装，无降级）**：`requirements.txt` 全部依赖已通过
**清华镜像**安装并验证可导入：scipy 1.18.1 / scikit-learn 1.9.1 / xgboost 3.4.1 /
lightgbm 4.7.0 / catboost 1.2.10 / statsmodels 0.15.0。当前 `output/` 全部由
`python run.py` 在**真实三模型**下重跑生成，主模型为 `xgboost`；见
`output/walk_forward_log.json`（三模型 `early_stopping` + F1 权重）、
`output/classification_report.json`（四路对照）与 `output/self_check.json`
（后端状态「全部可用」）。`src/preprocess.py` / `src/models.py` 中的纯 numpy 降级路径
仅在依赖缺失时自动启用并登记为偏离，**当前交付未触发降级**。

**环境复算记录：降级运行 → 真实 XGBoost**

| 指标 | 降级运行（历史留存） | 真实 XGBoost（当前交付） | 变化 |
|---|---|---|---|
| 年化收益 | 16.46% | 15.25% | −1.21pct |
| 夏普 | 2.02 | 1.87 | −0.15 |
| 最大回撤 | 4.84% | 5.28% | +0.44pct |
| Precision(类1+2) | 0.5169 | 0.4810 | −0.0359 |
| Recall(类1+2) | 0.7931 | 0.6552 | −0.1379 |
| F1(类1+2) | 0.6259 | 0.5547 | −0.0712 |

降级线性 softmax 偏高的 recall 来自近似恒定的做多倾向，不代表模型能力；换成真实三模型后
分类质量与净值表现均回落。持仓暴露仍达 95.06%、平均持有 38.5 个交易日，说明
「再现买点重置持有期」造成的长持有问题源于交易规则口径，与模型后端无关。

输出文件：

| 文件 | 内容 |
|---|---|
| `output/nav.csv` | 逐日净值、持仓、信号、基准 |
| `output/trades.csv` | 每笔平仓交易 |
| `output/metrics.json` | 绩效指标 |
| `output/classification_report.json` | 各模型/集成的 Precision/Recall/F1 |
| `output/comparison_to_paper.csv` | 与原文数值差异表 |
| `output/sensitivity.csv` | 持仓期/MAD/滚动窗口/种子敏感性 |
| `output/self_check.json` | 无未来函数、成本一致性、依赖、缓存自检 |
| `output/preprocess_report.json` | 剔除列、填补策略、Box-Cox、KS 明细 |
| `output/walk_forward_log.json` | 逐年重训记录与集成权重 |
| `output/segment_a_report.json` | 段 A 独立模型对照 |
| `output/summary.md` | 完整策略摘要与偏离登记 |

---

## 5. 结果验证方式

1. **一键复算**：`python run.py` 后核对 `output/metrics.json` 与 `output/nav.csv`；
   随机种子固定为 42，重复运行数值应一致。
2. **无未来函数**：`output/self_check.json` 的因子扰动测试 —— 扰动 t 之后的收盘价，
   t 及之前的因子必须完全不变；标签用扩展窗口分位数；walk-forward 训练显式排除样本外标签。
3. **口径一致性**：`config.yaml` 为唯一配置源；自检校验成本参数、依赖、缓存完整性。
4. **参数敏感性**：`output/sensitivity.csv` 观察绩效是否只在单点参数上成立（过拟合信号）。
5. **对照**：`comparison_to_paper.csv`（与原文数值）、`classification_report.json`（各模型对照）、
   `segment_a_report.json`（段 A 独立模型）。
6. **边界**：不含实盘下单、不连交易接口；结果仅供研究与教育。

---

## 6. 已知偏离与风险（方法学复现登记）

| 项 | 原文 | 本次 |
|---|---|---|
| 预测目标 | 30Y 国债收益率方向 | 期货收盘价代理（`S0059752` 无权限） |
| 数据范围 | 约 100 因子、过去 10 年 | 4 个 EDB 指标 + 价格衍生因子；段 A 缓存从 2022-01-04 起 |
| 标签 | 全样本历史分位数；上行样本未定义 | 扩展窗口分位数；上行样本并入类别 0 |
| 建模 | 三模型 + 动态集成（XGB 最优） | 两段各训一个模型；XGB 为主，其余与集成作对照 |
| 成交价 | 未写明 | 当日收盘建仓（**已知轻微前视偏差**） |
| 成本 | 不考虑滑点/杠杆/保证金 | 0/0/0，未做成本敏感性（按用户选择） |
| 容量 | 未提及 | 未评估，登记为风险项 |
| 预处理 | 阈值未给 | 固定默认值；MAD 改为训练集全局统计（原文按年度会泄漏）；KS 15 年按 9 年折算 |
| 指标口径 | 未写明 | 夏普 rf=0/252、回撤基于净值、胜率分母为平仓交易 |

已知风险：轻微前视、零成本、因子子集、缓存区间短于原文、两种 Wind 价格接口口径差异、
样本外区间行情与原文不同。以上均可能使结果偏乐观或不可与原文直接比较。

---

## 7. 合规声明

投资有风险，本内容仅供研究与教育，不构成投资建议；回测结果不等于实盘表现，
历史业绩不代表未来。不荐股、不给买卖点与目标价、不承诺收益。
涨跌表述按国内习惯：红涨绿跌。涉税涉法请转专业人士。
