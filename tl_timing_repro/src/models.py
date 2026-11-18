"""模型层：XGBoost / LightGBM / CatBoost + F1 加权动态集成。

对应确认口径：库默认超参、随机种子 42、验证集 20% early stopping、
训练集对类别 1/2 上采样至平衡、不做贝叶斯搜索、按年滚动重训 walk-forward。
主模型为 XGBoost，其余模型与集成均输出作对照。
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    _HAS_SKLEARN = True
except Exception:  # 纯 numpy 兜底，保证离线可运行
    _HAS_SKLEARN = False

    def precision_score(yt, yp, zero_division=0):
        yt, yp = np.asarray(yt).astype(int), np.asarray(yp).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        pp = int((yp == 1).sum())
        return float(tp / pp) if pp else float(zero_division)

    def recall_score(yt, yp, zero_division=0):
        yt, yp = np.asarray(yt).astype(int), np.asarray(yp).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum())
        ap = int((yt == 1).sum())
        return float(tp / ap) if ap else float(zero_division)

    def f1_score(yt, yp, zero_division=0):
        p = precision_score(yt, yp, zero_division)
        r = recall_score(yt, yp, zero_division)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else float(zero_division)


warnings.filterwarnings("ignore")

CLASSES = np.array([0, 1, 2])


class _NumpySoftmax:
    """第三方 GBDT 库不可用时的降级模型：纯 numpy 多项 softmax 回归。

    仅用于保证流水线可运行与结果可复算；会在输出报告中标记为降级模型，
    不代表原文/XGBoost 的模型效果。
    """

    def __init__(self, seed: int = 42, lr: float = 0.08, epochs: int = 400):
        self.seed, self.lr, self.epochs = seed, lr, epochs
        self.classes_ = CLASSES.copy()

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0)
        self.sd_[self.sd_ < 1e-9] = 1.0
        Z = np.clip((X - self.mu_) / self.sd_, -8, 8)
        Z = np.c_[np.ones(len(Z)), Z]
        Y = np.zeros((len(y), 3))
        for k in range(3):
            Y[y == k, k] = 1.0
        rng = np.random.RandomState(self.seed)
        W = rng.normal(0, 0.01, size=(Z.shape[1], 3))
        for _ in range(self.epochs):
            logits = np.clip(Z @ W, -30, 30)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            P = e / e.sum(axis=1, keepdims=True)
            W -= self.lr * (Z.T @ (P - Y) / len(y) + 1e-4 * W)
        self.W_ = W
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Z = np.c_[np.ones(len(X)), np.clip((X - self.mu_) / self.sd_, -8, 8)]
        logits = np.clip(Z @ self.W_, -30, 30)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _make_model(name: str, cfg: dict, seed: int):
    n = int(cfg["model"]["n_estimators"])
    es = int(cfg["model"]["early_stopping_rounds"])
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=n, random_state=seed, eval_metric="mlogloss",
            tree_method="hist", early_stopping_rounds=es,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=n, random_state=seed, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=n, random_seed=seed, verbose=0,
            allow_writing_files=False, loss_function="MultiClass",
        )
    raise ValueError(f"未知模型: {name}")


def upsample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """随机复制类别 1/2 样本，使三类样本量相同。"""
    rng = np.random.RandomState(seed)
    counts = {c: int((y == c).sum()) for c in CLASSES}
    target = max(counts.values())
    parts_X, parts_y = [X], [y]
    for c in (1, 2):
        idx = np.where(y == c)[0]
        need = target - len(idx)
        if need > 0 and len(idx) > 0:
            take = rng.choice(idx, size=need, replace=True)
            parts_X.append(X[take])
            parts_y.append(y[take])
    Xo = np.concatenate(parts_X, axis=0)
    yo = np.concatenate(parts_y, axis=0)
    order = rng.permutation(len(yo))
    return Xo[order], yo[order]


def _align_proba(model, proba: np.ndarray) -> np.ndarray:
    """把模型输出的概率列对齐到类别 0/1/2。"""
    out = np.zeros((proba.shape[0], 3), dtype=float)
    classes = np.asarray(getattr(model, "classes_", CLASSES))
    for j, c in enumerate(classes):
        k = int(c)
        if 0 <= k <= 2:
            out[:, k] = proba[:, j]
    return out


def _fit_one(name: str, X: np.ndarray, y: np.ndarray, cfg: dict, seed: int):
    model = _make_model(name, cfg, seed)
    frac = float(cfg["model"]["validation_frac"])
    n_val = max(1, int(len(y) * frac))
    X_tr, y_tr = X[:-n_val], y[:-n_val]
    X_val, y_val = X[-n_val:], y[-n_val:]
    if len(np.unique(y_tr)) < 2:
        X_tr, y_tr, X_val, y_val = X, y, X[-n_val:], y[-n_val:]
    note = "early_stopping"
    try:
        if name == "xgboost":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        elif name == "lightgbm":
            from lightgbm import early_stopping
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[early_stopping(int(cfg["model"]["early_stopping_rounds"]), verbose=False)])
        else:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=int(cfg["model"]["early_stopping_rounds"]))
    except Exception as exc:  # 不同库版本参数差异 -> 降级为固定轮数
        note = f"early_stopping 降级为固定轮数: {type(exc).__name__}"
        try:
            model = _make_model(name, cfg, seed)
            if name == "xgboost":
                model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
        except Exception:
            model.fit(X, y)
    return model, note


def train_models(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int) -> dict:
    """训练全部模型，返回模型、F1 权重与训练备注。"""
    Xu, yu = upsample(X, y, seed) if cfg["label"]["upsample"] else (X, y)
    models, notes, weights = {}, {}, {}
    for name in cfg["model"]["models"]:
        try:
            model, note = _fit_one(name, Xu, yu, cfg, seed)
        except ImportError as exc:
            notes[name] = f"库未安装，已跳过: {exc}"
            continue
        except Exception as exc:
            notes[name] = f"训练失败，已跳过: {type(exc).__name__}: {exc}"
            continue
        models[name] = model
        notes[name] = note
        try:
            pred = model.predict(X).astype(int)
            f1 = f1_score((y > 0).astype(int), (pred > 0).astype(int), zero_division=0)
        except Exception:
            f1 = 0.0
        weights[name] = max(float(f1), 1e-6)
    if not models:
        # 所有第三方 GBDT 库都不可用：降级为内置 softmax，保证流水线可运行
        sm = _NumpySoftmax(seed).fit(Xu, yu)
        models["builtin_softmax"] = sm
        notes["builtin_softmax"] = "第三方 GBDT 库不可用，降级为纯 numpy softmax 多分类（已登记偏离）"
        pred = sm.predict(X).astype(int)
        weights["builtin_softmax"] = max(
            float(f1_score((y > 0).astype(int), (pred > 0).astype(int), zero_division=0)), 1e-6
        )
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}
    return {"models": models, "notes": notes, "weights": weights}


def predict_all(bundle: dict, X: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    for name, model in bundle["models"].items():
        out[name] = _align_proba(model, np.asarray(model.predict_proba(X)))
    if out:
        ens = np.zeros_like(next(iter(out.values())))
        for name, p in out.items():
            ens += bundle["weights"][name] * p
        out["ensemble"] = ens
    return out


def proba_to_label(p: np.ndarray) -> np.ndarray:
    return np.asarray(p).argmax(axis=1).astype(int)


def binary12_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """类别 1+2 合并为「做多信号」后的 precision / recall / F1。"""
    yt = (np.asarray(y_true) > 0).astype(int)
    yp = (np.asarray(y_pred) > 0).astype(int)
    return {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "n_samples": int(len(yt)),
        "n_positive_true": int(yt.sum()),
        "n_positive_pred": int(yp.sum()),
    }


def walk_forward_predict(
    X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex,
    cfg: dict, seed: int, predict_mask: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """按年滚动重训：对每个含待预测样本的年份，只用该年之前的数据训练。

    返回各模型/集成在 predict_mask 上的预测，以及重训记录。
    """
    years = pd.Series(dates.year, index=range(len(dates)))
    filled: Dict[str, np.ndarray] = {}
    log = []
    y_arr = y.to_numpy(dtype=float)
    valid_train = (~np.isnan(y_arr)) & (~predict_mask)  # 绝不使用待预测样本的标签
    for yr in sorted(set(years.to_numpy()[predict_mask])):
        test_idx = np.where(predict_mask & (years.to_numpy() == yr))[0]
        train_idx = np.where((years.to_numpy() < yr) & valid_train)[0]
        if len(test_idx) == 0:
            continue
        if len(train_idx) < 50:  # 样本太少时使用所有「非待预测」的有效样本
            train_idx = np.where(valid_train)[0]
            note = "该年之前样本不足，使用全部训练期有效样本（仍排除样本外）"
        else:
            note = "按年滚动重训"
        bundle = train_models(
            X.iloc[train_idx].to_numpy(dtype=float),
            y.iloc[train_idx].to_numpy(dtype=int), cfg, seed,
        )
        preds = predict_all(bundle, X.iloc[test_idx].to_numpy(dtype=float))
        for name, p in preds.items():
            if name not in filled:
                filled[name] = np.full((len(dates), 3), np.nan)
            filled[name][test_idx] = p
        log.append({
            "year": int(yr), "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)), "note": note,
            "weights": bundle["weights"], "model_notes": bundle["notes"],
        })
    return filled, {"folds": log}
