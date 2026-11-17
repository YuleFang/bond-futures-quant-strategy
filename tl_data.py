"""
tl_data.py — 30 年期国债期货（TL）择时策略的数据层

数据源优先级：
  1) Wind 终端（WindPy）：用于生产级完整复现，字段清单见 WIND_FIELD_SPECS；
  2) 公开数据（AkShare，中债估值中心 / 中国货币网 / 新浪财经 / 中金所行情镜像）：
     当前环境无 Wind 数据权限，实际回测走这条链路，数据全部为真实历史行情，
     并缓存到 tl_data_panel.csv 以便离线复跑。

输出：日频面板 tl_data_panel.csv + 特征矩阵 tl_features.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PANEL_CSV = HERE / "tl_data_panel.csv"
FEATURE_CSV = HERE / "tl_features.csv"

DATA_START = "2015-01-01"
DATA_END = "2026-11-11"          # 当前可得数据的最后交易日，随运行环境更新
FORWARD_DAYS = 5

# ---------------------------------------------------------------------------
# Wind 字段清单（代码经 Wind EDB 指标检索核验；期货行情字段为 Wind 标准 mnemonic）
# ---------------------------------------------------------------------------
# (中文名, Wind 代码/接口, 字段, 频率, 用途)
WIND_FIELD_SPECS = [
    ("中债国债到期收益率:30年", "S0059752 / w.edb", "close", "日", "预测目标、标签构造、久期方向"),
    ("中债国债到期收益率:10年", "S0059749 / w.edb", "close", "日", "期限利差、曲线斜率"),
    ("中债国债到期收益率:1年", "S0059744 / w.edb", "close", "日", "短端锚、资金-短债剪刀差"),
    ("国债收益率曲线 3M/6M/3Y/5Y/7Y/30Y", "w.edb 曲线族 / w.wset('yieldcurve')", "close", "日", "曲线形态、蝶式、水平/斜率因子"),
    ("DR007", "M1006337 / w.edb", "close", "日", "银行间资金价格、杠杆成本"),
    ("银行间质押式回购加权平均利率:1天", "M0010073 / w.edb", "close", "月", "资金面低频校验（日频用 FR001/FDR001 替代）"),
    ("FR001/FR007/FDR001/FDR007", "w.edb 回购定盘利率族", "close", "日", "资金面、跨市场背离"),
    ("中债中短期票据收益率曲线(AAA) 3Y/5Y", "w.edb 曲线族", "close", "日", "信用利差、风险偏好"),
    ("CPI当月同比 / PPI全部工业品当月同比", "w.edb 宏观族", "close", "月", "通胀基本面（对齐交易日、滞后发布）"),
    ("制造业PMI / 工业增加值同比 / 社融存量同比 / M2同比", "w.edb 宏观族", "close", "月", "经济增长与广义流动性"),
    ("债券净融资额 / 中债托管量", "w.edb 债券族", "close", "日/月", "债市供求"),
    ("沪深300指数收盘价 / 市盈率(TTM)", "000300.SH / w.wsd", "close,pe_ttm", "日", "股债相对回报率、情绪面"),
    ("一线城市地铁客运量 / 粗钢日均产量 / SCFI / 水泥价格指数", "w.edb 高频族", "close", "周/旬", "高频基本面"),
    ("30年期国债期货主力连续", "TL.CFE / w.wsd", "open,high,low,close,volume,oi,settle", "日", "交易标的与技术面因子"),
    ("30年期国债期货具体月份合约", "TLxxxx.CFE / w.wsd", "close,oi,settle", "日", "换月拼接、主力连续校正"),
]


def _wind_loader():  # pragma: no cover - 需要本地 Wind 终端与数据权限
    """生产环境用：WindPy 取数模板（代码/字段需在 Wind 终端按账号权限核验）。"""
    from WindPy import w  # noqa

    w.start()
    edb_codes = "S0059744,S0059749,S0059752,M1006337"
    raw = w.edb(edb_codes, DATA_START.replace("-", ""), DATA_END.replace("-", ""), "Fill=Previous")
    yld = pd.DataFrame(np.array(raw.Data).T, index=raw.Times, columns=raw.Codes)
    fut = w.wsd("TL.CFE", "open,high,low,close,volume,oi", DATA_START, DATA_END, "PriceAdj=F")
    tl = pd.DataFrame(np.array(fut.Data).T, index=fut.Times, columns=[c.lower() for c in fut.Fields])
    panel = yld.join(tl, how="outer")
    panel.index.name = "date"
    return panel.reset_index()


# ---------------------------------------------------------------------------
# 公开数据链路
# ---------------------------------------------------------------------------
def _year_ranges(start: str, end: str):
    """返回 [(YYYYMMDD, YYYYMMDD), ...]，统一格式，避免 '20260101' 与
    '2026-11-11' 字符串比较导致最后一年被误跳过。"""
    out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(f"{y}0101"), pd.Timestamp(start))
        e = min(pd.Timestamp(f"{y}1231"), pd.Timestamp(end))
        if s > e:
            continue
        out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
    return out


def _fetch_yield_curve() -> pd.DataFrame:
    import akshare as ak

    frames = []
    for s, e in _year_ranges(DATA_START, DATA_END):
        for attempt in range(3):
            try:
                chunk = ak.bond_china_yield(start_date=s, end_date=e)
                if len(chunk):
                    frames.append(chunk)
                    break
            except Exception as exc:  # 年度请求失败则短暂等待重试
                if attempt == 2:
                    print(f"[warn] 收益率曲线 {s}~{e} 取数失败: {exc}")
                else:
                    time.sleep(1)
        time.sleep(0.15)
    raw = pd.concat(frames, ignore_index=True)

    keep = {
        "中债国债收益率曲线": "g",
        "中债中短期票据收益率曲线(AAA)": "aaa",
        "中债商业银行普通债收益率曲线(AAA)": "bank",
    }
    out = []
    tenors = ["3月", "6月", "1年", "3年", "5年", "7年", "10年", "30年"]
    for name, prefix in keep.items():
        sub = raw.loc[raw["曲线名称"] == name, ["日期"] + tenors].copy()
        sub["日期"] = pd.to_datetime(sub["日期"])
        sub = sub.drop_duplicates("日期").set_index("日期")
        sub = sub.rename(columns={t: f"{prefix}_{t}" for t in tenors})
        out.append(sub)
    curve = pd.concat(out, axis=1)
    curve.index.name = "date"
    return curve.reset_index()


def _fetch_repo() -> pd.DataFrame:
    import akshare as ak

    frames = []
    for s, e in _year_ranges(DATA_START, DATA_END):
        try:
            chunk = ak.repo_rate_hist(start_date=s, end_date=e)
            frames.append(chunk)
        except Exception as exc:
            print(f"[warn] 回购定盘利率 {s}~{e} 取数失败: {exc}")
        time.sleep(0.15)
    repo = pd.concat(frames, ignore_index=True)
    repo["date"] = pd.to_datetime(repo["date"])
    return repo.drop_duplicates("date").set_index("date").sort_index()


def _fetch_equity() -> pd.DataFrame:
    import akshare as ak

    hs = ak.stock_zh_index_daily(symbol="sh000300").copy()
    hs["date"] = pd.to_datetime(hs["date"])
    hs = hs.drop_duplicates("date").set_index("date").sort_index()
    return hs[["close", "volume"]].rename(columns={"close": "hs300", "volume": "hs300_vol"})


def _fetch_tl() -> pd.DataFrame:
    import akshare as ak

    tl = ak.futures_zh_daily_sina(symbol="TL0").copy()
    tl["date"] = pd.to_datetime(tl["date"])
    tl = tl.drop_duplicates("date").set_index("date").sort_index()
    return tl.rename(columns={
        "open": "tl_open", "high": "tl_high", "low": "tl_low",
        "close": "tl_close", "volume": "tl_volume", "hold": "tl_oi",
        "settle": "tl_settle",
    })


def load_panel(refresh: bool = False, source: str = "public") -> pd.DataFrame:
    if PANEL_CSV.exists() and not refresh and source == "public":
        df = pd.read_csv(PANEL_CSV, parse_dates=["date"])
        return df

    if source == "wind":
        panel = _wind_loader()
    else:
        print("[data] 拉取中债国债收益率曲线 / 信用债曲线 ...")
        curve = _fetch_yield_curve()
        print("[data] 拉取银行间回购定盘利率 ...")
        repo = _fetch_repo()
        print("[data] 拉取沪深300与TL主力连续日线 ...")
        hs = _fetch_equity()
        tl = _fetch_tl()
        panel = (curve.set_index("date")
                 .join(repo, how="outer")
                 .join(hs, how="outer")
                 .join(tl, how="outer")
                 .sort_index())
        panel = panel.loc[panel.index >= pd.Timestamp(DATA_START)]
        panel.index.name = "date"
        panel = panel.reset_index()

    panel.to_csv(PANEL_CSV, index=False, encoding="utf-8-sig")
    print(f"[data] 面板已缓存: {PANEL_CSV}  shape={panel.shape}, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel


# ---------------------------------------------------------------------------
# 特征工程（全部只用 t 及 t 之前的信息）
# ---------------------------------------------------------------------------
def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 低频序列按交易日前向填充（宏观数据发布滞后的近似；严禁向后填充）
    for col in df.columns:
        if col != "date" and col.startswith(("g_", "aaa_", "bank_", "FR", "FDR", "hs300")):
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill()

    y30 = df["g_30年"].astype(float)
    y10 = df["g_10年"].astype(float)
    y5 = df["g_5年"].astype(float)
    y3 = df["g_3年"].astype(float)
    y1 = df["g_1年"].astype(float)

    f: dict[str, pd.Series] = {}

    # 1) 趋势 / 动量：利率 5/10/20 日变动，均线偏离，波动率，历史分位
    for k in (1, 3, 5, 10, 20, 60):
        f[f"dy30_{k}"] = y30.diff(k)
        f[f"dy10_{k}"] = y10.diff(k)
    for w in (5, 10, 20, 60):
        ma = y30.rolling(w).mean()
        f[f"y30_dev_ma{w}"] = y30 - ma
    f["y30_z252"] = (y30 - y30.rolling(252).mean()) / y30.rolling(252).std()
    f["y30_pct252"] = y30.rolling(252).rank(pct=True)
    for w in (20, 60):
        f[f"vol_dy30_{w}"] = y30.diff().rolling(w).std()
    f["dy30_accel"] = f["dy30_5"] - f["dy30_5"].shift(5)

    # 2) 曲线结构：期限利差及其变动、蝶式
    sp = {
        "sp30_10": y30 - y10,
        "sp10_1": y10 - y1,
        "sp5_1": y5 - y1,
        "sp30_5": y30 - y5,
        "sp10_5": y10 - y5,
        "fly_1_5_10": 2 * y5 - y1 - y10,
        "fly_5_10_30": 2 * y10 - y5 - y30,
    }
    for name, series in sp.items():
        f[name] = series
        f[f"{name}_d5"] = series.diff(5)
        f[f"{name}_d20"] = series.diff(20)

    # 3) 资金面 / 流动性：回购定盘利率、资金-短债剪刀差、跨市场背离
    for col in ("FR001", "FR007", "FDR001", "FDR007"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            f[col.lower()] = s
            f[f"{col.lower()}_d5"] = s.diff(5)
            f[f"{col.lower()}_ma20_dev"] = s - s.rolling(20).mean()
    f["carry_spread"] = f.get("fdr007", pd.Series(np.nan, index=df.index)) - y1
    f["repo_basis"] = f.get("fr007", pd.Series(np.nan, index=df.index)) - f.get("fdr007", pd.Series(np.nan, index=df.index))

    # 4) 情绪 / 相对价值：信用利差、股债相对回报、股票波动
    for t in ("3年", "5年", "10年"):
        if f"aaa_{t}" in df.columns:
            cs = pd.to_numeric(df[f"aaa_{t}"], errors="coerce") - pd.to_numeric(df[f"g_{t}"], errors="coerce")
            key = f"cs_aaa_{t}"
            f[key] = cs
            f[f"{key}_d5"] = cs.diff(5)
            f[f"{key}_pct252"] = cs.rolling(252).rank(pct=True)
    if "hs300" in df.columns:
        hs = pd.to_numeric(df["hs300"], errors="coerce")
        for k in (1, 5, 20, 60):
            f[f"hs300_r{k}"] = hs.pct_change(k) * 100
        f["hs300_vol20"] = hs.pct_change().rolling(20).std() * 100
        # 股债相对回报：股票 20 日收益 - 10Y 利率下行带来的近似债券资本利得（久期≈10）
        f["eq_bond_rel"] = f["hs300_r20"] - (-f["dy10_20"] / 100 * 10 * 100)

    # 5) TL 期货技术面（2023-04-21 上市前为 NaN，后续标准化后填 0）
    if "tl_close" in df.columns:
        c = pd.to_numeric(df["tl_close"], errors="coerce")
        h = pd.to_numeric(df["tl_high"], errors="coerce")
        l = pd.to_numeric(df["tl_low"], errors="coerce")
        o = pd.to_numeric(df["tl_open"], errors="coerce")
        v = pd.to_numeric(df["tl_volume"], errors="coerce")
        oi = pd.to_numeric(df["tl_oi"], errors="coerce")
        for k in (1, 3, 5, 10, 20):
            f[f"tl_r{k}"] = c.pct_change(k) * 100
        for w in (5, 10, 20):
            f[f"tl_dev_ma{w}"] = (c / c.rolling(w).mean() - 1) * 100
        f["tl_range"] = (h - l) / c * 100
        f["tl_gap"] = (o / c.shift(1) - 1) * 100
        f["tl_vol20"] = c.pct_change().rolling(20).std() * 100
        f["tl_dvol5"] = np.log1p(v) - np.log1p(v).rolling(5).mean()
        f["tl_doi5"] = oi - oi.rolling(5).mean()
        tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        f["tl_atr14"] = tr.rolling(14).mean() / c * 100
        delta = c.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        dn = (-delta.clip(upper=0)).rolling(14).mean()
        f["tl_rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        f["tl_close_pos"] = (c - l) / (h - l).replace(0, np.nan) * 100

    # 6) 日历
    f["month"] = df["date"].dt.month
    f["dom"] = df["date"].dt.day

    feat = pd.DataFrame(f, index=df.index)
    # 剔除常数列与全空列
    feat = feat.loc[:, feat.notna().any()]
    feat = feat.loc[:, feat.nunique(dropna=True) > 3]

    # 滚动稳健标准化（日频 1 年窗口，等价于原文的 (X - Median)/IQR）
    med = feat.rolling(252, min_periods=120).median()
    q75 = feat.rolling(252, min_periods=120).quantile(0.75)
    q25 = feat.rolling(252, min_periods=120).quantile(0.25)
    iqr = (q75 - q25).replace(0, np.nan)
    z = ((feat - med) / iqr).clip(-5, 5)
    # 上市前 / 数据缺口按"中性值 0"填充，避免树模型对缺失方向产生伪规律
    z = z.fillna(0.0)
    z.columns = [f"z_{c}" for c in z.columns]

    out = pd.concat([df[["date"]], z], axis=1)
    for c in ("g_30年", "g_10年", "g_1年", "tl_open", "tl_high", "tl_low", "tl_close"):
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- 标签：未来 5 个交易日的 30Y 收益率变动（仅用于训练/评估，绝不进特征） ----
    # 【前视偏差说明】下方 shift(-5) 是唯一一处读取未来数据的代码，且只用于构造
    # 监督学习的目标变量 label / fwd_dy30_5。特征列全部以 z_ 前缀，train_and_predict
    # 只取 z_ 开头的列，label 与 fwd_* / q* 均被 FEATURE_EXCLUDE 与列名前缀双重排除；
    # 因此 t 时刻的预测输入不包含任何 t 之后的信息。
    dy_fwd = y30.shift(-FORWARD_DAYS) - y30
    # 历史分位阈值：用截至 t 已实现的 5 日变动的滚动分位数，严格无未来信息
    dy_past = y30.diff(FORWARD_DAYS)
    q20 = dy_past.rolling(756, min_periods=252).quantile(0.20)
    q40 = dy_past.rolling(756, min_periods=252).quantile(0.40)
    q60 = dy_past.rolling(756, min_periods=252).quantile(0.60)
    label = np.where(dy_fwd <= q20, 2, np.where(dy_fwd <= q40, 1, 0))
    out["fwd_dy30_5"] = dy_fwd
    out["q20"] = q20
    out["q40"] = q40
    out["q60"] = q60
    out["label"] = pd.Series(label, index=df.index).astype(float)
    out.loc[dy_fwd.isna() | q20.isna(), "label"] = np.nan
    out["label_long"] = out["label"].isin([1.0, 2.0]).astype(int)
    out.loc[out["label"].isna(), "label_long"] = np.nan

    out.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
    print(f"[features] {FEATURE_CSV}  shape={out.shape}, 特征数={len(z.columns)}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--source", choices=["public", "wind"], default="public")
    args = ap.parse_args()
    p = load_panel(refresh=args.refresh, source=args.source)
    fx = build_features(p)
    print(fx[["date", "g_30年", "tl_close", "label"]].tail(10).to_string())
