"""数据层：读取 Wind 导出的本地缓存，并保留 Wind 取数接口留桩。

数据边界（用户确认）：**只用万得**。本地缓存即本会话通过 Wind MCP 工具取得并
落地的真实数据，来源与口径见 ``data/DATA_MANIFEST.md``。

三种数据源
----------
1. ``cache``（默认）：读取 ``data/*.csv``，离线可运行、结果可复算。
2. ``mcp``：留桩。Wind MCP 工具只能在 WorkBuddy 会话内调用，导出流程见
   ``tools/import_wind_results.py``；在会话中重新取数后重跑该脚本即可扩充缓存。
3. ``windpy``：留桩。若本机装有 Wind 终端与 WindPy，可用 ``w.edb`` / ``w.wsd``
   自行补全 2015-05-14 起的全量数据，落地为同名 CSV 后重跑主程序。

任何情况下都不使用 Akshare 或其他外部数据源，也不生成/伪造行情数据。
"""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(BASE_DIR, "config.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _data_dir(cfg: dict) -> str:
    return os.path.join(BASE_DIR, cfg["data"]["cache_dir"])


def load_prices(cfg: dict, segment: str = "segment_b") -> pd.DataFrame:
    """读取段 A / 段 B 的日 K，统一列名并按日期排序。"""
    spec = cfg["data"][segment]
    path = os.path.join(_data_dir(cfg), spec["file"])
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates("date", keep="last")
    df["segment"] = segment
    df["code"] = spec["code"]
    return df


def load_edb(cfg: dict) -> Dict[str, pd.Series]:
    """读取 EDB 指标为以日期为索引的 Series（保留原始频率，不做对齐）。"""
    out: Dict[str, pd.Series] = {}
    for code, spec in cfg["data"]["edb"].items():
        path = os.path.join(_data_dir(cfg), spec["file"])
        s = pd.read_csv(path, parse_dates=["date"])
        s = s.drop_duplicates("date", keep="last").set_index("date")["value"]
        out[code] = s.sort_index().astype(float)
    return out


def align_edb_to_trading_days(
    edb: Dict[str, pd.Series], trade_dates: pd.Series, cfg: dict
) -> pd.DataFrame:
    """把不同频率的 EDB 指标对齐到交易日：仅使用 <= t 的已发布观测，前向填充。

    注意：EDB 的 date 为数据日期，前向填充到交易日是常规做法；若该指标的
    date 实为发布日，则存在轻微前视，已在 README 偏离登记中标注。
    """
    idx = pd.DatetimeIndex(pd.to_datetime(trade_dates))
    frame = pd.DataFrame(index=idx)
    for code, s in edb.items():
        frame[code] = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    return frame


# ---------------------------------------------------------------------------
# Wind 取数留桩
# ---------------------------------------------------------------------------

WIND_FIELD_SPEC = {
    "price": {
        "tool": "get_index_kline",
        "codes": ["T.CFE", "TL.CFE"],
        "fields": ["TIME", "OPEN", "MATCH(收盘)", "HIGH", "LOW", "VOLUME"],
        "frequency": "日 K (period=10)",
        "note": "主力连续；get_financial_data 返回的是另一套价格口径，勿混用",
    },
    "edb": {
        "tool": "natural_language_get_edb_data (仅提数)",
        "codes": {
            "M1006337": "DR007",
            "M0041652": "R001 银行间质押式回购加权利率:1天",
            "S5914515": "中国:水泥价格指数",
            "S0114089": "SCFI:综合指数",
            "S5708246": "中国:日均产量:粗钢:重点企业（已验证可取，未纳入本次缓存）",
            "S0059749": "中债国债到期收益率:10年（权限未验证）",
            "S0059752": "中债国债到期收益率:30年（**当前账号无权限**）",
        },
        "frequency": "按指标原始频率，交易日后对齐",
    },
}


def describe_wind_fields() -> str:
    """返回所需字段与数据范围说明（供 README / 报告引用）。"""
    lines = ["### 所需 Wind 字段与范围", ""]
    p = WIND_FIELD_SPEC["price"]
    lines.append(f"- 价量：`{p['tool']}`，代码 {p['codes']}，字段 {p['fields']}，{p['frequency']}")
    lines.append(f"  - {p['note']}")
    lines.append(f"- 宏观/流动性：`{WIND_FIELD_SPEC['edb']['tool']}`，{WIND_FIELD_SPEC['edb']['frequency']}")
    for code, name in WIND_FIELD_SPEC["edb"]["codes"].items():
        lines.append(f"  - `{code}` {name}")
    lines.append("- 目标口径范围：段 A 2015-05-14~2023-04-20；段 B 2023-04-21~2025-05-14；"
                 "样本外 2024-05-14~2025-05-14")
    lines.append("- 本地缓存实际范围：段 A 2022-01-04~2023-04-20；段 B 2023-04-21~2025-05-14")
    return "\n".join(lines)


def fetch_from_windpy(*args, **kwargs):  # pragma: no cover - 留桩
    raise NotImplementedError(
        "WindPy 后端需要本机 Wind 终端与 WindPy。请自行用 w.wsd/w.edb 取数并落地为 "
        "data/prices_*.csv 与 data/edb_*.csv，然后重跑 python run.py（默认 cache 源）。"
    )


def fetch_from_wind_mcp(*args, **kwargs):  # pragma: no cover - 留桩
    raise NotImplementedError(
        "Wind MCP 只能在 WorkBuddy 会话内通过工具调用；取数后运行 "
        "python tools/import_wind_results.py 导入缓存。"
    )


if __name__ == "__main__":
    print(describe_wind_fields())
