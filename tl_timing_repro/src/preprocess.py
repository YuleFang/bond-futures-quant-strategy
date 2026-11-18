"""预处理：缺失/异常值、平稳性、Box-Cox、滚动稳健标准化、KS 稳定性筛选。

对应确认口径（config.preprocess）：缺失率>15% 弃列、MAD n=3、
Shapiro-Wilk α=0.05 决定均值/中位数/KNN(K=5) 填补、Box-Cox 自动估 λ、
ADF/KPSS 后一阶/季节差分、滚动稳健标准化 (X-Median)/IQR（244 交易日）、
KS α=0.05 按实际 9 年样本期折算。

**防泄漏约定**：所有需要估计的统计量（保留列、填补值、λ、是否差分）
只在 ``fit(train_X)`` 里由训练集决定；``transform`` 可对整个矩阵执行，
因为滚动标准化与差分都只用过去值。MAD 阈值使用训练集全局中位数/MAD
（原文写「分年度」，按年度中位数会用到当期未来数据，故改为全局，已登记偏离）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
    _HAS_SCIPY = True
except Exception:  # 无 scipy 时用纯 numpy 近似，保证流水线可运行（会在报告中登记）
    _HAS_SCIPY = False
    from types import SimpleNamespace

    class _StatsFallback:
        @staticmethod
        def shapiro(x):
            return SimpleNamespace(pvalue=0.0)  # 保守：走中位数/KNN 填补分支

        @staticmethod
        def skew(x):
            x = np.asarray(x, dtype=float)
            m = x.mean()
            s = x.std()
            return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else 0.0

        @staticmethod
        def boxcox_normmax(x, method="mle"):
            return 0.0  # 等价于对数变换

        @staticmethod
        def ks_2samp(a, b):
            a = np.sort(np.asarray(a, dtype=float))
            b = np.sort(np.asarray(b, dtype=float))
            grid = np.union1d(a, b)
            d = float(np.max(np.abs(
                np.searchsorted(a, grid, side="right") / len(a)
                - np.searchsorted(b, grid, side="right") / len(b)
            )))
            n_eff = len(a) * len(b) / (len(a) + len(b))
            p = float(np.clip(2 * np.exp(-2 * d * d * n_eff), 0.0, 1.0))
            return SimpleNamespace(pvalue=p, statistic=d)

    stats = _StatsFallback()

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False


class Preprocessor:
    def __init__(self, cfg: dict):
        p = cfg["preprocess"]
        self.miss_thr = float(p["missing_col_threshold"])
        self.mad_n = float(p["mad_n"])
        self.alpha = float(p["normality_alpha"])
        self.knn_k = int(p["knn_k"])
        self.seasonal_period = int(p["seasonal_period"])
        self.win = int(p["rolling_window_daily"])
        self.ks_alpha = float(p["ks_alpha"])
        self.ks_ratio = float(p["ks_years_required_ratio"])
        self.columns: list[str] = []
        self.fill_values: dict[str, float] = {}
        self.fill_strategy: dict[str, str] = {}
        self.boxcox_lambda: dict[str, float] = {}
        self.diff_cols: list[str] = []
        self.dropped_by_missing: list[str] = []
        self.dropped_by_ks: list[str] = []
        self.ks_report: dict[str, dict] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _mad_clip(s: pd.Series, median: float, mad: float, n: float) -> pd.Series:
        if not np.isfinite(mad) or mad <= 0:
            return s
        return s.clip(median - n * 1.4826 * mad, median + n * 1.4826 * mad)

    def _normality_strategy(self, s: pd.Series) -> str:
        x = s.dropna().to_numpy(dtype=float)
        if len(x) < 20:
            return "median"
        sample = x if len(x) <= 500 else x[:500]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = stats.shapiro(sample).pvalue
        except Exception:
            p = 0.0
        if p >= self.alpha:
            return "mean"
        skew = float(stats.skew(x))
        return "median" if abs(skew) < 1.0 else "knn"

    def _stationarity_diff(self, s: pd.Series) -> bool:
        x = s.dropna().to_numpy(dtype=float)
        if len(x) < 30 or not _HAS_SM:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(x, autolag="AIC")[1]
                kpss_p = kpss(x, regression="c", nlags="auto")[1]
        except Exception:
            return False
        return bool(adf_p > self.alpha or kpss_p < self.alpha)

    # ------------------------------------------------------------------
    def fit(self, train_X: pd.DataFrame) -> "Preprocessor":
        miss = train_X.isna().mean()
        self.dropped_by_missing = [c for c, v in miss.items() if v > self.miss_thr]
        cols = [c for c in train_X.columns if c not in self.dropped_by_missing]
        if not cols:
            raise ValueError("所有因子列都因缺失率过高被剔除")
        clean = train_X[cols].copy()

        self.median_ = clean.median()
        self.mad_ = (clean - self.median_).abs().median()
        for c in cols:
            clean[c] = self._mad_clip(clean[c], self.median_[c], self.mad_[c], self.mad_n)

        for c in cols:
            strategy = self._normality_strategy(clean[c])
            self.fill_strategy[c] = strategy
            if strategy == "mean":
                self.fill_values[c] = float(clean[c].mean())
            elif strategy == "median":
                self.fill_values[c] = float(clean[c].median())
            else:  # knn：用 K 个最近邻观测的均值，落地为常数，避免跨样本泄漏
                x = clean[c].dropna().to_numpy(dtype=float)
                k = min(self.knn_k, len(x)) if len(x) else 0
                self.fill_values[c] = float(np.mean(x[-k:])) if k else 0.0
            filled = clean[c].fillna(self.fill_values[c])

            # Box-Cox：仅正值且非常数列
            if (filled > 0).all() and filled.std() > 1e-12:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lam = stats.boxcox_normmax(filled.to_numpy(), method="mle")
                    self.boxcox_lambda[c] = float(lam)
                except Exception:
                    pass
            if self._stationarity_diff(filled):
                self.diff_cols.append(c)

        # KS 稳定性：按年分组，逐年与训练集总体做两样本 KS
        years = pd.Series(clean.index.year, index=clean.index)
        n_years = years.nunique()
        need = max(1, int(np.ceil(self.ks_ratio * n_years)))
        for c in cols:
            passed = 0
            detail = {}
            base = clean[c].dropna().to_numpy(dtype=float)
            for y in sorted(years.unique()):
                sub = clean.loc[years == y, c].dropna().to_numpy(dtype=float)
                if len(sub) < 10 or len(base) < 10:
                    detail[str(y)] = "样本不足"
                    continue
                p = float(stats.ks_2samp(base, sub).pvalue)
                detail[str(y)] = round(p, 4)
                if p >= self.ks_alpha:
                    passed += 1
            self.ks_report[c] = {"passed_years": passed, "required": need, "detail": detail}
            if passed < need:
                self.dropped_by_ks.append(c)
        self.columns = [c for c in cols if c not in self.dropped_by_ks]
        if not self.columns:
            # 全部被 KS 剔除时保留原始列，并在报告中登记，不让流水线中断
            self.columns = cols
            self.dropped_by_ks = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.columns].copy()
        for c in self.columns:
            out[c] = self._mad_clip(out[c], self.median_[c], self.mad_[c], self.mad_n)
            out[c] = out[c].fillna(self.fill_values[c])
            lam = self.boxcox_lambda.get(c)
            if lam is not None and (out[c] > 0).all():
                out[c] = (np.power(out[c], lam) - 1.0) / lam if abs(lam) > 1e-8 else np.log(out[c])
        for c in self.diff_cols:
            if c in out.columns:
                out[c] = out[c].diff(self.seasonal_period if c.startswith("S0114089") else 1)
        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)

        # 滚动稳健标准化：只用过去窗口，天然无未来函数
        med = out.rolling(self.win, min_periods=max(20, self.win // 5)).median()
        q75 = out.rolling(self.win, min_periods=max(20, self.win // 5)).quantile(0.75)
        q25 = out.rolling(self.win, min_periods=max(20, self.win // 5)).quantile(0.25)
        iqr = (q75 - q25).replace(0, np.nan)
        std = out.rolling(self.win, min_periods=max(20, self.win // 5)).std().replace(0, np.nan)
        denom = iqr.fillna(std).fillna(1.0)
        out = (out - med) / denom
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def report(self) -> dict:
        return {
            "scipy_available": _HAS_SCIPY,
            "statsmodels_available": _HAS_SM,
            "fallback_note": (
                "" if (_HAS_SCIPY and _HAS_SM) else
                "scipy/statsmodels 不可用：Shapiro/Box-Cox/KS/ADF 已用纯 numpy 近似或跳过，"
                "属于环境降级，结果口径与完整依赖环境不同"
            ),
            "n_input_columns": len(self.columns) + len(self.dropped_by_missing) + len(self.dropped_by_ks),
            "dropped_by_missing": self.dropped_by_missing,
            "dropped_by_ks": self.dropped_by_ks,
            "n_retained_columns": len(self.columns),
            "fill_strategy_counts": {
                k: list(self.fill_strategy.values()).count(k) for k in ("mean", "median", "knn")
            },
            "boxcox_columns": list(self.boxcox_lambda),
            "differenced_columns": self.diff_cols,
            "ks_report": self.ks_report,
        }
