# Wind 数据缓存清单

- 生成时间：2026-11-17T14:12:34
- 生成方式：`python tools/import_wind_results.py`，来源为 WorkBuddy 持久化的 Wind MCP 工具结果
- 段 A（T.CFE）：314 个交易日，2022-01-04 ~ 2023-04-20
- 段 B（TL.CFE）：497 个交易日，2023-04-21 ~ 2025-05-14
- 价格主口径：`get_index_kline` 的 MATCH(收盘)/OPEN/HIGH/LOW/VOLUME；Wind 代码 T.CFE / TL.CFE（主力连续）
- 已排除 `get_financial_data` 返回的另一套 TL.CFE 价格口径，避免混用

## EDB 指标

| 代码 | 行数 | 起始 | 结束 |
|---|---|---|---|
| M0041652 | 837 | 2022-01-04 | 2025-05-14 |
| M1006337 | 837 | 2022-01-04 | 2025-05-14 |
| S0114089 | 167 | 2022-01-07 | 2025-05-09 |
| S5914515 | 827 | 2022-01-04 | 2025-05-14 |

## 来源文件

| 文件 | 类型 | 行数 |
|---|---|---|
| chatcmpl-tool-848ab3cdd9a2867b.txt | EDB 时间序列 | 2668 |
| chatcmpl-tool-89cf630e9ba4228a-render.txt | get_index_kline | 237 |
| chatcmpl-tool-8f308d5ec9f4249e-render.txt | get_index_kline | 314 |
| chatcmpl-tool-a0649605b4b6f560-render.txt | get_index_kline | 260 |
