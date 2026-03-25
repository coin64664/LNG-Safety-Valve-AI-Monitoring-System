from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from risk_engine import AI_FEATURES, AI_FEATURE_LABELS, RISK_HIGH, RISK_SAFE, RISK_WARN, compute_mechanistic_health, risk_from_hi

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def _robust_score(value: float, center: float, spread: float, scale: float = 18.0) -> float:
    if pd.isna(value) or pd.isna(center):
        return 0.0
    if pd.isna(spread) or float(spread) == 0:
        spread = 1.0
    return float(abs((float(value) - float(center)) / float(spread)) * scale)


def _group_windows(group: pd.DataFrame, pos: int, window_days: int) -> pd.DataFrame:
    ordered = group.sort_values("date")
    return ordered.iloc[max(0, pos - (window_days - 1)) : pos + 1].copy()


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

            med = win[feature_cols].median(numeric_only=True)
            q90 = win[feature_cols].quantile(0.90, numeric_only=True)
            q10 = win[feature_cols].quantile(0.10, numeric_only=True)
            band = (q90 - q10).replace(0, np.nan)

            ratio_score = _robust_score(scored.at[idx, "ratio"], med.get("ratio", np.nan), band.get("ratio", np.nan))
            pmax_score = _robust_score(scored.at[idx, "p_max"], med.get("p_max", np.nan), band.get("p_max", np.nan))
            temp_score = _robust_score(scored.at[idx, "temp"], med.get("temp", np.nan), band.get("temp", np.nan), scale=10.0)
            level_score = _robust_score(scored.at[idx, "level"], med.get("level", np.nan), band.get("level", np.nan), scale=10.0)
            slope_score = _robust_score(scored.at[idx, "slope_3d"], med.get("slope_3d", np.nan), band.get("slope_3d", np.nan), scale=16.0)
            jump_score = _robust_score(scored.at[idx, "recent_jump"], med.get("recent_jump", np.nan), band.get("recent_jump", np.nan), scale=16.0)
            vol_score = _robust_score(scored.at[idx, "volatility_3d"], med.get("volatility_3d", np.nan), band.get("volatility_3d", np.nan), scale=16.0)
            activity_score = _robust_score(scored.at[idx, "Activity"], med.get("Activity", np.nan), band.get("Activity", np.nan), scale=12.0)

            scored.at[idx, "baseline_dev_score"] = float(np.clip(np.mean([ratio_score, pmax_score, temp_score, level_score, slope_score, jump_score, vol_score, activity_score]), 0, 100))
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
    return scored


def build_ai_reason_top_features(row: pd.Series) -> tuple[str, str, str]:
    reasons = {
        "接近整定压力": max(0.0, (float(row.get("ratio", 0)) - 0.94) * 120),
        "近3日最高压力持续上升": max(0.0, float(row.get("continuous_rise_days", 0)) * 12 + float(row.get("slope_3d", 0)) * 500),
        "压力波动放大": max(0.0, float(row.get("volatility_3d", 0)) * 1200),
        "短时压力突增": max(0.0, float(row.get("recent_jump", 0)) * 1400),
        "温度偏离本站历史基线": max(0.0, abs(float(row.get("temp", 0)) - float(row.get("baseline_temp_median", row.get("temp", 0)))) * 2.2),
        "液位偏离本站历史基线": max(0.0, abs(float(row.get("level", 0)) - float(row.get("baseline_level_median", row.get("level", 0)))) * 0.8),
        "动作/微放散异常": max(0.0, float(row.get("Activity", 0)) * 20),
        "相对历史基线偏离明显": max(0.0, float(row.get("baseline_dev_score", 0))),
    }
    top = [name for name, _ in sorted(reasons.items(), key=lambda item: item[1], reverse=True) if _ > 0][:3]
    while len(top) < 3:
        top.append("-")
    return top[0], top[1], top[2]


def compute_iforest_signals(df: pd.DataFrame, contamination: float, window_days: int = 60, min_samples: int = 30) -> pd.DataFrame:
    scored = df.copy()
    for col, default in {
        "ai_raw_score": np.nan,
        "ai_score_pct": np.nan,
        "ai_observe_flag": False,
        "ai_escalate_flag": False,
        "AI_anomaly": False,
        "AI_score": np.nan,
    }.items():
        scored[col] = default

    if len(scored) == 0 or not SKLEARN_OK:
        return scored

    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        ordered_idx = list(ordered.index)
        valid_idx = []

        for pos, idx in enumerate(ordered_idx):
            win = _group_windows(ordered, pos, window_days)
            xw = win[AI_FEATURES].apply(pd.to_numeric, errors="coerce")
            xw = xw.fillna(xw.median(numeric_only=True)).fillna(0.0)
            if len(xw) < min_samples:
                continue

            scaler = StandardScaler()
            xs = scaler.fit_transform(xw.values)
            model = IsolationForest(n_estimators=300, contamination=float(contamination), random_state=42)
            model.fit(xs)

            raw_score = float(-model.score_samples(xs[-1].reshape(1, -1))[0])
            scored.at[idx, "ai_raw_score"] = raw_score
            valid_idx.append(idx)

        if not valid_idx:
            continue

        pct = pd.to_numeric(scored.loc[valid_idx, "ai_raw_score"], errors="coerce").rank(pct=True, method="average") * 100
        scored.loc[valid_idx, "ai_score_pct"] = pct
        scored.loc[valid_idx, "ai_observe_flag"] = scored.loc[valid_idx, "ai_score_pct"] >= 95
        scored.loc[valid_idx, "AI_anomaly"] = scored.loc[valid_idx, "ai_observe_flag"]
        scored.loc[valid_idx, "AI_score"] = scored.loc[valid_idx, "ai_raw_score"]

    return scored


def _build_observe_streak(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["observe_streak"] = 0
    trigger_series = (
        scored["ai_observe_flag"].fillna(False)
        | (pd.to_numeric(scored["baseline_dev_score"], errors="coerce").fillna(0) >= 60)
        | (pd.to_numeric(scored["degradation_score"], errors="coerce").fillna(0) >= 55)
    )
    scored["observe_seed"] = trigger_series
    for _, group in scored.groupby(["station", "valve_type"], sort=False):
        ordered = group.sort_values("date")
        streak = 0
        for idx in ordered.index:
            streak = streak + 1 if bool(scored.at[idx, "observe_seed"]) else 0
            scored.at[idx, "observe_streak"] = streak
    return scored


def build_risk_reason_path(row: pd.Series) -> str:
    stage = row.get("risk_stage", "正常")
    if not bool(row.get("baseline_ready", False)):
        return "当前样本不足，系统仅依据机理健康指数进行规则判断，暂不输出自适应AI升级结论。"

    reasons = [row.get("ai_reason_top1", "-"), row.get("ai_reason_top2", "-"), row.get("ai_reason_top3", "-")]
    reasons = [item for item in reasons if item and item != "-"]
    if not reasons:
        reasons = ["整体运行基本稳定"]

    if stage == "AI高风险":
        return f"近阶段出现 {'、'.join(reasons[:3])}，且异常已形成持续性与共识增强，系统判定为高风险状态。"
    if stage == "AI升级":
        return f"检测到 {'、'.join(reasons[:3])}，并与规则风险或持续异常共同出现，系统判定为退化升级风险。"
    if stage == "AI观察":
        return f"检测到 {'、'.join(reasons[:2])}，系统进入观察状态，建议加强巡检和趋势复核。"
    return "当前未发现显著异常共识，系统判定总体运行平稳。"


def compute_consensus_risk(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = _build_observe_streak(df.copy())
    baseline = pd.to_numeric(scored["baseline_dev_score"], errors="coerce").fillna(0)
    degradation = pd.to_numeric(scored["degradation_score"], errors="coerce").fillna(0)
    ai_pct = pd.to_numeric(scored["ai_score_pct"], errors="coerce").fillna(0)
    rule_penalty = np.where(scored["Risk"] == RISK_WARN, 10, 0) + np.where(scored["Risk"] == RISK_HIGH, 22, 0)
    scored["consensus_score"] = np.clip(baseline * 0.34 + degradation * 0.36 + ai_pct * 0.20 + rule_penalty, 0, 100)

    observe = (
        scored["ai_observe_flag"].fillna(False)
        | (baseline >= 60)
        | (degradation >= 55)
    )
    escalate = (
        scored["ai_escalate_flag"].fillna(False)
        | ((observe) & (scored["observe_streak"] >= 2))
        | ((scored["Risk"].isin([RISK_WARN, RISK_HIGH])) & observe)
        | (baseline >= 78)
        | (degradation >= 72)
    )
    high = (
        (escalate)
        & (
            (degradation >= 85)
            | (scored["ratio"] >= 1.00)
            | (scored["near_set_days"] >= 2)
            | (scored["observe_streak"] >= 3)
            | (scored["consensus_score"] >= 82)
        )
    )

    scored["risk_stage"] = "正常"
    scored.loc[observe, "risk_stage"] = "AI观察"
    scored.loc[escalate, "risk_stage"] = "AI升级"
    scored.loc[high, "risk_stage"] = "AI高风险"

    penalty = observe.astype(int) * 4 + escalate.astype(int) * 8 + high.astype(int) * 14
    penalty += np.clip(degradation * 0.12 + baseline * 0.08, 0, 24)
    scored["HI_final"] = np.clip(scored["HI"] - penalty, 0, 100)
    scored["Risk_final"] = scored["HI_final"].apply(risk_from_hi)
    scored.loc[(observe) & (scored["Risk_final"] == RISK_SAFE), "Risk_final"] = RISK_WARN
    scored.loc[high, "Risk_final"] = RISK_HIGH

    t1, t2, t3 = zip(*scored.apply(build_ai_reason_top_features, axis=1))
    scored["ai_reason_top1"] = list(t1)
    scored["ai_reason_top2"] = list(t2)
    scored["ai_reason_top3"] = list(t3)
    scored["risk_reason_path"] = scored.apply(build_risk_reason_path, axis=1)
    scored["ai_escalate_flag"] = escalate.astype(bool)
    return scored


def run_scoring_pipeline(df: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    scored = compute_mechanistic_health(df)
    scored["baseline_dev_score"] = np.nan
    scored["degradation_score"] = 0.0
    scored["consensus_score"] = 0.0
    scored["risk_stage"] = "正常"
    scored["risk_reason_path"] = "当前仅完成规则健康评估。"
    scored["ai_raw_score"] = np.nan
    scored["ai_score_pct"] = np.nan
    scored["ai_observe_flag"] = False
    scored["ai_escalate_flag"] = False
    scored["ai_reason_top1"] = "-"
    scored["ai_reason_top2"] = "-"
    scored["ai_reason_top3"] = "-"
    scored["AI_anomaly"] = False
    scored["AI_score"] = np.nan
    scored["baseline_ready"] = False
    scored["HI_final"] = scored["HI"]
    scored["Risk_final"] = scored["Risk"]

    if enable_ai and SKLEARN_OK:
        scored = compute_adaptive_baseline(scored)
        scored = compute_degradation_score(scored)
        scored = compute_iforest_signals(scored, contamination=contamination, window_days=60, min_samples=30)
        scored = compute_consensus_risk(scored)

    return scored


def build_validation_summary(df_scored: pd.DataFrame, alerts_df: pd.DataFrame) -> dict:
    if df_scored is None or len(df_scored) == 0:
        return {
            "group_count": 0,
            "eligible_group_count": 0,
            "observe_count": 0,
            "escalate_count": 0,
            "high_count": 0,
            "close_rate": 0.0,
        }

    grouped = df_scored.groupby(["station", "valve_type"]).size().reset_index(name="samples")
    eligible = grouped[grouped["samples"] >= 30]
    data_source_mix = df_scored.get("data_source_tag", pd.Series(dtype=str)).value_counts().to_dict()
    close_rate = float((alerts_df["status"] == "已关闭").mean() * 100) if alerts_df is not None and len(alerts_df) else 0.0
    return {
        "group_count": int(len(grouped)),
        "eligible_group_count": int(len(eligible)),
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
        "max_consensus_score": float(pd.to_numeric(subset["consensus_score"], errors="coerce").max()),
        "avg_hi_final": float(pd.to_numeric(subset["HI_final"], errors="coerce").mean()),
        "last7_avg_hi": float(pd.to_numeric(last7["HI_final"], errors="coerce").mean()) if len(last7) else np.nan,
        "observe_days": int(subset["risk_stage"].isin(["AI观察", "AI升级", "AI高风险"]).sum()),
        "upgrade_days": int(subset["risk_stage"].isin(["AI升级", "AI高风险"]).sum()),
    }
