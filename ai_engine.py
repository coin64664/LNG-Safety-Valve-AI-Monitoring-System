from __future__ import annotations

from datetime import timedelta
from typing import Dict

import numpy as np
import pandas as pd

from risk_engine import RISK_HIGH, RISK_SAFE, RISK_WARN, compute_mechanistic_health, risk_from_hi

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

MODEL_LABELS = {
    "if": "Isolation Forest",
    "lof": "局部离群因子",
    "shift": "时序突变检测",
    "degradation": "退化趋势引擎",
}


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _robust_score(value: float, center: float, spread: float, scale: float = 18.0) -> float:
    if pd.isna(value) or pd.isna(center):
        return 0.0
    if pd.isna(spread) or float(spread) == 0:
        spread = 1.0
    return float(abs((float(value) - float(center)) / float(spread)) * scale)


def _group_windows(group: pd.DataFrame, pos: int, window_days: int) -> pd.DataFrame:
    ordered = group.sort_values("date")
    return ordered.iloc[max(0, pos - (window_days - 1)) : pos + 1].copy()


def _normalize_by_active(component_map: Dict[str, tuple[float, float, bool]]) -> float:
    score = 0.0
    weight = 0.0
    for _, (value, cur_weight, ready) in component_map.items():
        if not ready or pd.isna(value):
            continue
        score += float(value) * cur_weight
        weight += cur_weight
    if weight == 0:
        return 0.0
    return float(score / weight)


def compute_adaptive_baseline(df: pd.DataFrame, window_days: int = 60, min_samples: int = 7) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = df.copy().sort_values(["station", "valve_type", "date"]).reset_index(drop=True)
    baseline_cols = [
        "baseline_dev_score",
        "baseline_pmax_median",
        "baseline_ratio_median",
        "baseline_temp_median",
        "baseline_level_median",
        "baseline_ready",
    ]
    for col in baseline_cols:
        scored[col] = np.nan if col != "baseline_ready" else False

    feature_cols = ["p_max", "ratio", "temp", "level", "slope_3d", "recent_jump", "volatility_3d", "Activity"]

    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        ordered_idx = list(ordered.index)
        for pos, idx in enumerate(ordered_idx):
            win = _group_windows(ordered, pos, window_days)
            if len(win) < min_samples:
                continue

            history = win.iloc[:-1].copy() if len(win) > min_samples else win.copy()
            if len(history) < min_samples:
                history = win.copy()

            med = history[feature_cols].median(numeric_only=True)
            q90 = history[feature_cols].quantile(0.90, numeric_only=True)
            q10 = history[feature_cols].quantile(0.10, numeric_only=True)
            band = (q90 - q10).replace(0, np.nan)

            ratio_score = _robust_score(scored.at[idx, "ratio"], med.get("ratio", np.nan), band.get("ratio", np.nan))
            pmax_score = _robust_score(scored.at[idx, "p_max"], med.get("p_max", np.nan), band.get("p_max", np.nan))
            temp_score = _robust_score(scored.at[idx, "temp"], med.get("temp", np.nan), band.get("temp", np.nan), scale=10.0)
            level_score = _robust_score(scored.at[idx, "level"], med.get("level", np.nan), band.get("level", np.nan), scale=10.0)
            slope_score = _robust_score(scored.at[idx, "slope_3d"], med.get("slope_3d", np.nan), band.get("slope_3d", np.nan), scale=16.0)
            jump_score = _robust_score(scored.at[idx, "recent_jump"], med.get("recent_jump", np.nan), band.get("recent_jump", np.nan), scale=16.0)
            vol_score = _robust_score(scored.at[idx, "volatility_3d"], med.get("volatility_3d", np.nan), band.get("volatility_3d", np.nan), scale=16.0)
            activity_score = _robust_score(scored.at[idx, "Activity"], med.get("Activity", np.nan), band.get("Activity", np.nan), scale=12.0)

            baseline_score = np.mean([ratio_score, pmax_score, temp_score, level_score, slope_score, jump_score, vol_score, activity_score])
            scored.at[idx, "baseline_dev_score"] = float(np.clip(baseline_score, 0, 100))
            scored.at[idx, "baseline_pmax_median"] = med.get("p_max", np.nan)
            scored.at[idx, "baseline_ratio_median"] = med.get("ratio", np.nan)
            scored.at[idx, "baseline_temp_median"] = med.get("temp", np.nan)
            scored.at[idx, "baseline_level_median"] = med.get("level", np.nan)
            scored.at[idx, "baseline_ready"] = True

    return scored


def compute_degradation_score(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = df.copy()
    score = np.zeros(len(scored), dtype=float)
    score += np.where(scored["slope_3d"] > 0.005, 10, 0)
    score += np.where(scored["slope_3d"] > 0.010, 12, 0)
    score += np.where(scored["continuous_rise_days"] >= 2, 12, 0)
    score += np.where(scored["continuous_rise_days"] >= 3, 10, 0)
    score += np.where(scored["near_set_days"] >= 1, 12, 0)
    score += np.where(scored["near_set_days"] >= 2, 14, 0)
    score += np.where(scored["recent_jump"] > 0.015, 12, 0)
    score += np.where(scored["recent_jump"] > 0.030, 8, 0)
    score += np.where(scored["volatility_3d"] > 0.010, 8, 0)
    score += np.where(scored["volatility_3d"] > 0.020, 10, 0)
    score += np.where(scored["Activity"] > 0, 18, 0)
    score += np.where((scored["temp"] >= 33) & (scored["ratio"] >= 0.95), 10, 0)
    scored["degradation_score"] = np.clip(score, 0, 100)
    scored["ai_degradation_score"] = scored["degradation_score"]
    return scored


def compute_shift_score(df: pd.DataFrame, window_days: int = 30, min_samples: int = 14) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = df.copy().sort_values(["station", "valve_type", "date"]).reset_index(drop=True)
    scored["ai_shift_score"] = np.nan
    scored["shift_ready"] = False
    scored["shift_vote"] = False

    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        ordered_idx = list(ordered.index)
        for pos, idx in enumerate(ordered_idx):
            win = _group_windows(ordered, pos, window_days)
            history = win.iloc[:-1].copy()
            if len(history) < min_samples:
                continue

            cur = scored.loc[idx]
            pmax_med = _safe_numeric(history["p_max"]).median()
            pmax_spread = (_safe_numeric(history["p_max"]).quantile(0.75) - _safe_numeric(history["p_max"]).quantile(0.25)) or 0.005
            ratio_med = _safe_numeric(history["ratio"]).median()
            ratio_spread = (_safe_numeric(history["ratio"]).quantile(0.75) - _safe_numeric(history["ratio"]).quantile(0.25)) or 0.005
            jump_med = _safe_numeric(history["recent_jump"]).median()
            jump_spread = (_safe_numeric(history["recent_jump"]).quantile(0.75) - _safe_numeric(history["recent_jump"]).quantile(0.25)) or 0.003
            vol_med = _safe_numeric(history["volatility_3d"]).median()
            vol_spread = (_safe_numeric(history["volatility_3d"]).quantile(0.75) - _safe_numeric(history["volatility_3d"]).quantile(0.25)) or 0.003
            slope_med = _safe_numeric(history["slope_3d"]).median()
            slope_spread = (_safe_numeric(history["slope_3d"]).quantile(0.75) - _safe_numeric(history["slope_3d"]).quantile(0.25)) or 0.003
            ewma_line = _safe_numeric(history["p_max"]).ewm(span=min(8, len(history)), adjust=False).mean().iloc[-1]

            pmax_shift = _robust_score(cur.get("p_max", np.nan), pmax_med, pmax_spread, scale=20.0)
            ratio_shift = _robust_score(cur.get("ratio", np.nan), ratio_med, ratio_spread, scale=18.0)
            jump_shift = _robust_score(cur.get("recent_jump", np.nan), jump_med, jump_spread, scale=16.0)
            vol_shift = _robust_score(cur.get("volatility_3d", np.nan), vol_med, vol_spread, scale=16.0)
            slope_shift = _robust_score(cur.get("slope_3d", np.nan), slope_med, slope_spread, scale=16.0)
            ewma_shift = _robust_score(cur.get("p_max", np.nan), ewma_line, pmax_spread, scale=20.0)

            shift_score = np.mean([pmax_shift, ratio_shift, jump_shift, vol_shift, slope_shift, ewma_shift])
            scored.at[idx, "ai_shift_score"] = float(np.clip(shift_score, 0, 100))
            scored.at[idx, "shift_ready"] = True

    scored.loc[scored["shift_ready"] & (_safe_numeric(scored["ai_shift_score"]).fillna(0) >= 65), "shift_vote"] = True
    return scored


def _assign_percentile(scored: pd.DataFrame, score_col: str, pct_col: str, ready_col: str, flag_col: str, threshold: float = 95.0) -> pd.DataFrame:
    scored[pct_col] = np.nan
    scored[flag_col] = False
    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        valid_idx = [idx for idx in ordered.index if bool(scored.at[idx, ready_col]) and pd.notna(scored.at[idx, score_col])]
        if not valid_idx:
            continue
        pct = _safe_numeric(scored.loc[valid_idx, score_col]).rank(pct=True, method="average") * 100
        scored.loc[valid_idx, pct_col] = pct
        scored.loc[valid_idx, flag_col] = scored.loc[valid_idx, pct_col] >= threshold
    return scored


def compute_multi_model_signals(df: pd.DataFrame, contamination: float, window_days: int = 60, min_samples: int = 30) -> pd.DataFrame:
    scored = df.copy().sort_values(["station", "valve_type", "date"]).reset_index(drop=True)
    defaults = {
        "ai_if_score": np.nan,
        "ai_if_pct": np.nan,
        "if_ready": False,
        "if_vote": False,
        "ai_lof_score": np.nan,
        "ai_lof_pct": np.nan,
        "lof_ready": False,
        "lof_vote": False,
        "ai_raw_score": np.nan,
        "ai_score_pct": np.nan,
        "ai_observe_flag": False,
        "ai_escalate_flag": False,
        "AI_anomaly": False,
        "AI_score": np.nan,
    }
    for col, default in defaults.items():
        scored[col] = default

    if len(scored) == 0 or not SKLEARN_OK:
        return scored

    feature_cols = [
        "p_now",
        "p_max",
        "ratio",
        "slope_3d",
        "level",
        "temp",
        "Activity",
        "recent_jump",
        "volatility_3d",
        "continuous_rise_days",
    ]

    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        ordered_idx = list(ordered.index)
        for pos, idx in enumerate(ordered_idx):
            win = _group_windows(ordered, pos, window_days)
            xw = win[feature_cols].apply(pd.to_numeric, errors="coerce")
            xw = xw.fillna(xw.median(numeric_only=True)).fillna(0.0)
            if len(xw) < min_samples:
                continue

            scaler = StandardScaler()
            xs = scaler.fit_transform(xw.values)

            iso = IsolationForest(n_estimators=300, contamination=float(contamination), random_state=42)
            iso.fit(xs)
            scored.at[idx, "ai_if_score"] = float(-iso.score_samples(xs[-1].reshape(1, -1))[0])
            scored.at[idx, "if_ready"] = True

            n_neighbors = max(5, min(20, len(xs) - 1))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=float(contamination))
            lof.fit_predict(xs)
            scored.at[idx, "ai_lof_score"] = float(-lof.negative_outlier_factor_[-1])
            scored.at[idx, "lof_ready"] = True

    scored = _assign_percentile(scored, "ai_if_score", "ai_if_pct", "if_ready", "if_vote")
    scored = _assign_percentile(scored, "ai_lof_score", "ai_lof_pct", "lof_ready", "lof_vote")

    scored["ai_raw_score"] = scored[["ai_if_score", "ai_lof_score"]].max(axis=1, skipna=True)
    scored["ai_score_pct"] = scored[["ai_if_pct", "ai_lof_pct"]].max(axis=1, skipna=True)
    return scored


def build_ai_reason_top_features(row: pd.Series) -> tuple[str, str, str]:
    reasons = {
        "接近整定压力": max(0.0, (float(row.get("ratio", 0)) - 0.94) * 120),
        "近3日最高压力持续上升": max(0.0, float(row.get("continuous_rise_days", 0)) * 12 + float(row.get("slope_3d", 0)) * 500),
        "压力波动放大": max(0.0, float(row.get("volatility_3d", 0)) * 1200),
        "短时压力突增": max(0.0, float(row.get("recent_jump", 0)) * 1400),
        "温度偏离历史基线": max(0.0, abs(float(row.get("temp", 0)) - float(row.get("baseline_temp_median", row.get("temp", 0)))) * 2.2),
        "液位偏离历史基线": max(0.0, abs(float(row.get("level", 0)) - float(row.get("baseline_level_median", row.get("level", 0)))) * 0.8),
        "动作或微放散异常": max(0.0, float(row.get("Activity", 0)) * 20),
        "相对历史基线偏离明显": max(0.0, float(row.get("baseline_dev_score", 0))),
    }
    top = [name for name, val in sorted(reasons.items(), key=lambda item: item[1], reverse=True) if val > 0][:3]
    while len(top) < 3:
        top.append("-")
    return top[0], top[1], top[2]


def _build_observe_streak(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["observe_streak"] = 0
    trigger_series = (
        scored[["if_vote", "lof_vote", "shift_vote", "degradation_vote"]].any(axis=1)
        | (_safe_numeric(scored["baseline_dev_score"]).fillna(0) >= 65)
    )
    scored["observe_seed"] = trigger_series
    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        streak = 0
        for idx in ordered.index:
            streak = streak + 1 if bool(scored.at[idx, "observe_seed"]) else 0
            scored.at[idx, "observe_streak"] = streak
    return scored


def build_action_suggestion(row: pd.Series) -> str:
    stage = str(row.get("risk_stage", "正常"))
    ratio = float(pd.to_numeric(pd.Series([row.get("ratio", np.nan)]), errors="coerce").iloc[0]) if pd.notna(row.get("ratio", np.nan)) else np.nan
    activity = float(pd.to_numeric(pd.Series([row.get("Activity", 0)]), errors="coerce").iloc[0] or 0)
    temp = float(pd.to_numeric(pd.Series([row.get("temp", np.nan)]), errors="coerce").iloc[0]) if pd.notna(row.get("temp", np.nan)) else np.nan

    if stage == "AI高风险":
        if pd.notna(ratio) and ratio >= 1.00:
            return "建议立即复核压力表与安全阀状态，安排现场检查并暂停非必要工况波动。"
        return "建议立即安排专项巡检，优先核查阀门密封性、整定压力和近期工况变化。"
    if stage == "AI升级":
        if activity > 0:
            return "建议尽快复盘动作或微放散记录，安排阀门密封与回座状态检查。"
        if pd.notna(temp) and temp >= 33:
            return "建议结合环境温度与液位变化复核压力波动原因，并提高巡检频次。"
        return "建议24小时内完成复检，重点核查压力接近整定值和持续上升趋势。"
    if stage == "AI观察":
        return "建议连续跟踪近3天压力趋势，关注是否继续上升或再次接近整定压力。"
    return "当前运行平稳，建议按既定频次巡检并持续记录关键参数。"


def build_risk_reason_path(row: pd.Series) -> str:
    stage = str(row.get("risk_stage", "正常"))
    reasons = [row.get("ai_reason_top1", "-"), row.get("ai_reason_top2", "-"), row.get("ai_reason_top3", "-")]
    reasons = [item for item in reasons if item and item != "-"]

    if not bool(row.get("baseline_ready", False)) and not bool(row.get("shift_ready", False)):
        return "当前历史数据较少，系统以机理健康和退化趋势监测为主，建议继续积累运行记录。"

    if not reasons:
        reasons = ["整体运行状态稳定"]

    reason_text = "、".join(reasons[:3])
    suggestion = build_action_suggestion(row)

    if stage == "AI高风险":
        return f"系统识别到{reason_text}，且多路模型同时指向异常增强，判定为高风险状态。{suggestion}"
    if stage == "AI升级":
        return f"系统识别到{reason_text}，并与规则风险或持续异常共同出现，风险已进入升级状态。{suggestion}"
    if stage == "AI观察":
        return f"系统识别到{reason_text}，当前建议重点观察趋势变化。{suggestion}"
    return suggestion


def compute_model_consensus(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = df.copy()
    scored["degradation_vote"] = _safe_numeric(scored["ai_degradation_score"]).fillna(0) >= 60
    scored["ai_vote_count"] = (
        scored[["if_vote", "lof_vote", "shift_vote", "degradation_vote"]]
        .fillna(False)
        .astype(int)
        .sum(axis=1)
    )

    confidence = []
    for _, row in scored.iterrows():
        component_map = {
            "if": (row.get("ai_if_pct", np.nan), 0.28, bool(row.get("if_ready", False))),
            "lof": (row.get("ai_lof_pct", np.nan), 0.22, bool(row.get("lof_ready", False))),
            "shift": (row.get("ai_shift_score", np.nan), 0.22, bool(row.get("shift_ready", False))),
            "degradation": (row.get("ai_degradation_score", np.nan), 0.28, True),
        }
        confidence.append(_normalize_by_active(component_map))
    scored["ai_confidence"] = np.clip(confidence, 0, 100)

    baseline = _safe_numeric(scored["baseline_dev_score"]).fillna(0)
    degradation = _safe_numeric(scored["ai_degradation_score"]).fillna(0)
    rule_penalty = np.where(scored["Risk"] == RISK_WARN, 10, 0) + np.where(scored["Risk"] == RISK_HIGH, 18, 0)
    scored["consensus_score"] = np.clip(scored["ai_confidence"] * 0.72 + baseline * 0.18 + rule_penalty, 0, 100)

    scored = _build_observe_streak(scored)

    observe = (
        (scored["ai_vote_count"] >= 1)
        | (baseline >= 65)
        | (degradation >= 58)
    )
    escalate = (
        (scored["ai_vote_count"] >= 2)
        | ((scored["Risk"].isin([RISK_WARN, RISK_HIGH])) & (scored["ai_vote_count"] >= 1))
        | ((observe) & (scored["observe_streak"] >= 2))
        | (degradation >= 75)
    )
    high = (
        (scored["ai_vote_count"] >= 3)
        | (((scored["ratio"] >= 1.00) | (scored["near_set_days"] >= 2)) & ((degradation >= 70) | (scored["ai_vote_count"] >= 2)))
        | ((scored["Risk"] == RISK_HIGH) & (scored["ai_vote_count"] >= 1))
    )

    scored["risk_stage"] = "正常"
    scored.loc[observe, "risk_stage"] = "AI观察"
    scored.loc[escalate, "risk_stage"] = "AI升级"
    scored.loc[high, "risk_stage"] = "AI高风险"

    penalty = observe.astype(int) * 4 + escalate.astype(int) * 8 + high.astype(int) * 14
    penalty += np.clip(scored["consensus_score"] * 0.10, 0, 14)
    scored["HI_final"] = np.clip(scored["HI"] - penalty, 0, 100)
    scored["Risk_final"] = scored["HI_final"].apply(risk_from_hi)
    scored.loc[(observe) & (scored["Risk_final"] == RISK_SAFE), "Risk_final"] = RISK_WARN
    scored.loc[high, "Risk_final"] = RISK_HIGH

    t1, t2, t3 = zip(*scored.apply(build_ai_reason_top_features, axis=1))
    scored["ai_reason_top1"] = list(t1)
    scored["ai_reason_top2"] = list(t2)
    scored["ai_reason_top3"] = list(t3)
    scored["action_suggestion"] = scored.apply(build_action_suggestion, axis=1)
    scored["risk_reason_path"] = scored.apply(build_risk_reason_path, axis=1)
    scored["ai_observe_flag"] = observe.astype(bool)
    scored["ai_escalate_flag"] = scored["risk_stage"].isin(["AI升级", "AI高风险"])
    scored["AI_anomaly"] = observe.astype(bool)
    scored["AI_score"] = scored["ai_confidence"]
    scored["ai_score_pct"] = scored[["ai_if_pct", "ai_lof_pct"]].max(axis=1, skipna=True)
    scored.loc[scored["ai_score_pct"].isna(), "ai_score_pct"] = scored.loc[scored["ai_score_pct"].isna(), "ai_shift_score"]
    return scored


def build_model_vote_matrix(df: pd.DataFrame, station: str, valve_type: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    subset = df[(df["station"] == station) & (df["valve_type"] == valve_type)].copy().sort_values("date")
    if len(subset) == 0:
        return pd.DataFrame()

    view = subset[
        [
            "date",
            "ai_if_score",
            "ai_if_pct",
            "if_vote",
            "ai_lof_score",
            "ai_lof_pct",
            "lof_vote",
            "ai_shift_score",
            "shift_vote",
            "ai_degradation_score",
            "degradation_vote",
            "ai_vote_count",
            "ai_confidence",
            "consensus_score",
            "risk_stage",
        ]
    ].copy()
    return view.sort_values("date", ascending=False).reset_index(drop=True)


def run_scoring_pipeline(df: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    scored = compute_mechanistic_health(df)
    defaults = {
        "baseline_dev_score": np.nan,
        "degradation_score": 0.0,
        "ai_degradation_score": 0.0,
        "consensus_score": 0.0,
        "risk_stage": "正常",
        "risk_reason_path": "当前以规则健康评估为主。",
        "action_suggestion": "当前运行平稳，建议按既定频次巡检并持续记录关键参数。",
        "ai_if_score": np.nan,
        "ai_if_pct": np.nan,
        "if_ready": False,
        "if_vote": False,
        "ai_lof_score": np.nan,
        "ai_lof_pct": np.nan,
        "lof_ready": False,
        "lof_vote": False,
        "ai_shift_score": np.nan,
        "shift_ready": False,
        "shift_vote": False,
        "ai_vote_count": 0,
        "ai_confidence": 0.0,
        "ai_raw_score": np.nan,
        "ai_score_pct": np.nan,
        "ai_observe_flag": False,
        "ai_escalate_flag": False,
        "ai_reason_top1": "-",
        "ai_reason_top2": "-",
        "ai_reason_top3": "-",
        "AI_anomaly": False,
        "AI_score": np.nan,
        "baseline_ready": False,
        "HI_final": scored["HI"],
        "Risk_final": scored["Risk"],
    }
    for col, default in defaults.items():
        scored[col] = default

    scored = compute_adaptive_baseline(scored)
    scored = compute_degradation_score(scored)
    scored = compute_shift_score(scored)

    if enable_ai and SKLEARN_OK:
        scored = compute_multi_model_signals(scored, contamination=contamination, window_days=60, min_samples=30)

    scored = compute_model_consensus(scored)
    return scored


def build_validation_summary(df_scored: pd.DataFrame, alerts_df: pd.DataFrame) -> dict:
    if df_scored is None or len(df_scored) == 0:
        return {
            "group_count": 0,
            "eligible_group_count": 0,
            "eligible_shift_group_count": 0,
            "observe_count": 0,
            "escalate_count": 0,
            "high_count": 0,
            "close_rate": 0.0,
        }

    grouped = df_scored.groupby(["station", "valve_type"]).size().reset_index(name="samples")
    eligible = grouped[grouped["samples"] >= 30]
    shift_eligible = grouped[grouped["samples"] >= 14]
    data_source_mix = df_scored.get("data_source_tag", pd.Series(dtype=str)).value_counts().to_dict()
    close_rate = float((alerts_df["status"] == "已关闭").mean() * 100) if alerts_df is not None and len(alerts_df) else 0.0
    return {
        "group_count": int(len(grouped)),
        "eligible_group_count": int(len(eligible)),
        "eligible_shift_group_count": int(len(shift_eligible)),
        "observe_count": int(df_scored["risk_stage"].eq("AI观察").sum()),
        "escalate_count": int(df_scored["risk_stage"].eq("AI升级").sum()),
        "high_count": int(df_scored["risk_stage"].eq("AI高风险").sum()),
        "close_rate": close_rate,
        "data_source_mix": data_source_mix,
        "date_start": str(df_scored["date"].min()),
        "date_end": str(df_scored["date"].max()),
    }


def build_case_replay(df: pd.DataFrame, station: str, valve_type: str) -> dict:
    if df is None or len(df) == 0:
        return {}

    subset = df[(df["station"] == station) & (df["valve_type"] == valve_type)].copy().sort_values("date")
    if len(subset) == 0:
        return {}

    focus = subset.sort_values(["consensus_score", "date"], ascending=[False, False]).iloc[0]
    last7 = subset[subset["date"] >= (subset["date"].max() - timedelta(days=6))]
    return {
        "station": station,
        "valve_type": valve_type,
        "sample_count": int(len(subset)),
        "date_start": str(subset["date"].min()),
        "date_end": str(subset["date"].max()),
        "focus_date": str(focus["date"]),
        "focus_stage": str(focus.get("risk_stage", "正常")),
        "focus_reason": str(focus.get("risk_reason_path", "")),
        "action_suggestion": str(focus.get("action_suggestion", "")),
        "max_consensus_score": float(_safe_numeric(subset["consensus_score"]).max()),
        "avg_hi_final": float(_safe_numeric(subset["HI_final"]).mean()),
        "last7_avg_hi": float(_safe_numeric(last7["HI_final"]).mean()) if len(last7) else np.nan,
        "observe_days": int(subset["risk_stage"].isin(["AI观察", "AI升级", "AI高风险"]).sum()),
        "upgrade_days": int(subset["risk_stage"].isin(["AI升级", "AI高风险"]).sum()),
        "vote_peak": int(_safe_numeric(subset["ai_vote_count"]).max()),
        "confidence_peak": float(_safe_numeric(subset["ai_confidence"]).max()),
    }
