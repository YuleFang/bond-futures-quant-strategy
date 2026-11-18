"""把本会话中 Wind MCP 工具返回的持久化结果导入为本地 CSV 缓存。

用途
----
WorkBuddy 会把体积较大的工具结果落盘到 projects/<project>/tool-results/*.txt。
本脚本读取这些文件，自动识别
  1) get_index_kline 返回的日 K（columns 含 TIME / MATCH / VOLUME）
  2) natural_language_get_edb_data 返回的 EDB 序列（meta + date + value）
并按已确认口径落地：

  - 段 A：T.CFE  2022-01-04 ~ 2023-04-20  -> data/prices_T.csv
  - 段 B：TL.CFE 2023-04-21 ~ 2025-05-14  -> data/prices_TL.csv
  - EDB：按指标代码分别落地              -> data/edb_<CODE>.csv

注意
----
get_financial_data 返回的 TL.CFE 序列是另一套价格口径（2024-05-14 收盘
106.56 vs get_index_kline 的 105.62），本脚本显式排除含 "Wind代码" 列的
结果，避免把两种口径混进同一份缓存。该差异会在 README 的偏离登记中记录。

运行：python tools/import_wind_results.py
仅依赖 Python 标准库。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime

TOOL_DIR = (
    r"C:\Users\Administrator\.workbuddy\projects"
    r"\c-Users-Administrator-WorkBuddy-实验室"
    r"\2c4c2cde-ee50-4d60-9b32-7a8ebf7b4c1a\tool-results"
)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

SEG_SPLIT = "2023-04-21"  # TL 上市日；之前归段 A(T)，之后归段 B(TL)
DATE_START = "2022-01-04"
DATE_END = "2025-05-14"


def load_payload(path: str):
    """读取工具结果文件，剥掉可能的 {"content": "<json string>"} 外壳。"""
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            text = fh.read().strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("content"), str):
        inner = obj["content"].strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            return None
    return obj


def norm_date(raw: str) -> str:
    """TIME 形如 2024-05-14T00:00:00.000+08:00；EDB 形如 20240514。"""
    raw = str(raw).strip()
    if "T" in raw:
        return raw[:10]
    if len(raw) == 8 and raw.isdigit():
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return raw[:10]


def collect() -> tuple[dict, dict, list]:
    kline: dict[str, dict] = {}      # date -> row dict
    edb: dict[str, dict] = {}        # code -> {date: value}
    sources: list[tuple[str, str, int]] = []

    if not os.path.isdir(TOOL_DIR):
        print(f"[ERROR] 工具结果目录不存在: {TOOL_DIR}", file=sys.stderr)
        return kline, edb, sources

    for name in sorted(os.listdir(TOOL_DIR)):
        if not name.endswith(".txt"):
            continue
        path = os.path.join(TOOL_DIR, name)
        payload = load_payload(path)
        if not isinstance(payload, dict):
            continue
        data = payload.get("data")
        if not isinstance(data, dict):
            continue

        # --- K 线 ---
        cols = data.get("columns")
        rows = data.get("rows")
        if isinstance(cols, list) and isinstance(rows, list):
            names = [c.get("name", "") if isinstance(c, dict) else str(c) for c in cols]
            if "Wind代码" in names:
                sources.append((name, "get_financial_data(不同价格口径，已排除)", 0))
                continue
            if "TIME" in names and "MATCH" in names:
                idx = {n: i for i, n in enumerate(names)}
                cnt = 0
                for r in rows:
                    if not isinstance(r, list) or len(r) < len(names):
                        continue
                    d = norm_date(r[idx["TIME"]])
                    if not (DATE_START <= d <= DATE_END):
                        continue
                    try:
                        rec = {
                            "date": d,
                            "open": float(r[idx["OPEN"]]),
                            "high": float(r[idx["HIGH"]]),
                            "low": float(r[idx["LOW"]]),
                            "close": float(r[idx["MATCH"]]),
                            "volume": float(r[idx["VOLUME"]]),
                        }
                    except (ValueError, TypeError, KeyError):
                        continue
                    kline[d] = rec  # 同日期以最后一次 get_index_kline 结果为准
                    cnt += 1
                sources.append((name, "get_index_kline", cnt))

        # --- EDB 时间序列 ---
        series = data.get("data")
        if isinstance(series, list):
            for item in series:
                if not isinstance(item, dict):
                    continue
                meta = item.get("meta") or {}
                code = meta.get("code")
                dates = item.get("date")
                values = item.get("value")
                if not code or not isinstance(dates, list) or not isinstance(values, list):
                    continue
                bucket = edb.setdefault(code, {})
                for d, v in zip(dates, values):
                    d = norm_date(d)
                    if DATE_START <= d <= DATE_END:
                        bucket[d] = v
            if any(
                isinstance(i, dict) and isinstance(i.get("date"), list)
                for i in series
            ):
                sources.append((name, "EDB 时间序列", sum(len(i.get("date") or []) for i in series if isinstance(i, dict))))

    return kline, edb, sources


def write_csv(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    kline, edb, sources = collect()

    seg_a = sorted((r for d, r in kline.items() if d < SEG_SPLIT), key=lambda r: r["date"])
    seg_b = sorted((r for d, r in kline.items() if d >= SEG_SPLIT), key=lambda r: r["date"])
    cols = ["date", "open", "high", "low", "close", "volume"]
    write_csv(os.path.join(DATA_DIR, "prices_T.csv"), cols, [[r[c] for c in cols] for r in seg_a])
    write_csv(os.path.join(DATA_DIR, "prices_TL.csv"), cols, [[r[c] for c in cols] for r in seg_b])

    edb_summary = []
    for code, series in sorted(edb.items()):
        rows = [[d, v] for d, v in sorted(series.items())]
        write_csv(os.path.join(DATA_DIR, f"edb_{code}.csv"), ["date", "value"], rows)
        edb_summary.append((code, len(rows), rows[0][0], rows[-1][0]))

    manifest = [
        "# Wind 数据缓存清单",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 生成方式：`python tools/import_wind_results.py`，来源为 WorkBuddy 持久化的 Wind MCP 工具结果",
        f"- 段 A（T.CFE）：{len(seg_a)} 个交易日，{seg_a[0]['date']} ~ {seg_a[-1]['date']}" if seg_a else "- 段 A：无数据",
        f"- 段 B（TL.CFE）：{len(seg_b)} 个交易日，{seg_b[0]['date']} ~ {seg_b[-1]['date']}" if seg_b else "- 段 B：无数据",
        "- 价格主口径：`get_index_kline` 的 MATCH(收盘)/OPEN/HIGH/LOW/VOLUME；Wind 代码 T.CFE / TL.CFE（主力连续）",
        "- 已排除 `get_financial_data` 返回的另一套 TL.CFE 价格口径，避免混用",
        "",
        "## EDB 指标",
        "",
        "| 代码 | 行数 | 起始 | 结束 |",
        "|---|---|---|---|",
    ]
    manifest += [f"| {c} | {n} | {s} | {e} |" for c, n, s, e in edb_summary]
    manifest += ["", "## 来源文件", "", "| 文件 | 类型 | 行数 |", "|---|---|---|"]
    manifest += [f"| {n} | {t} | {c} |" for n, t, c in sources]
    with open(os.path.join(DATA_DIR, "DATA_MANIFEST.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(manifest) + "\n")

    print(f"段 A  prices_T.csv : {len(seg_a)} 行  {seg_a[0]['date']} ~ {seg_a[-1]['date']}" if seg_a else "段 A 无数据")
    print(f"段 B  prices_TL.csv: {len(seg_b)} 行  {seg_b[0]['date']} ~ {seg_b[-1]['date']}" if seg_b else "段 B 无数据")
    for c, n, s, e in edb_summary:
        print(f"EDB   {c}: {n} 行  {s} ~ {e}")
    if not seg_a or not seg_b:
        print("[WARN] 段数据不完整，请检查工具结果目录", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
