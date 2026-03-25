import numpy as np
import pandas as pd

SET_P = 1.32

RISK_SAFE = "🟢 安全"
RISK_WARN = "🟡 预警"
RISK_HIGH = "🔴 高风险"

AI_FEATURES = [
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

AI_FEATURE_LABELS = {
    "p_now": "当前压力偏离",
    "p_max": "最高压力偏离",
    "ratio": "接近整定值偏离",
    "slope_3d": "近3日压力斜率偏离",
    "level": "液位偏离",
    "temp": "温度偏离",
    "Activity": "动作/微放散异常",
    "recent_jump": "短时压力突增",
    "volatility_3d": "压力波动放大",
    "continuous_rise_days": "连续上升趋势",
}


def risk_from_hi(value: float) -> str:
    if value >= 85:
        return RISK_SAFE
    if value >= 70:
        return RISK_WARN
    return RISK_HIGH


def _consecutive_counts(flag_series: pd.Series) -> pd.Series:
    streak = 0
    out = []
    for raw in flag_series.fillna(False).tolist():
        streak = streak + 1 if bool(raw) else 0
        out.append(streak)
    return pd.Series(out, index=flag_series.index)


def extract_domain_features(df: pd.DataFrame, set_p: float = SET_P) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = df.copy()
    scored = scored.sort_values(["station", "valve_type", "date"]).reset_index(drop=True)
    scored["date"] = pd.to_datetime(scored["date"], errors="coerce").dt.date
    scored["ratio"] = scored["p_max"] / float(set_p)
    scored["pressure_margin"] = float(set_p) - scored["p_max"]
    scored["pressure_delta"] = scored["p_max"] - scored["p_now"]
    scored["Activity"] = scored.get("psv_act", 0).fillna(0) + scored.get("psv_weeping", 0).fillna(0)

    grouped = scored.groupby(["station", "valve_type"], sort=False)
    scored["slope_3d"] = grouped["p_max"].apply(lambda s: s.diff().rolling(3, min_periods=2).mean()).reset_index(level=[0, 1], drop=True)
    scored["recent_jump"] = grouped["p_max"].diff().fillna(0)
    scored["volatility_3d"] = grouped["p_max"].apply(lambda s: s.diff().abs().rolling(3, min_periods=2).mean()).reset_index(level=[0, 1], drop=True)
    scored["pmax_rolling_mean_7"] = grouped["p_max"].apply(lambda s: s.rolling(7, min_periods=1).mean()).reset_index(level=[0, 1], drop=True)
    scored["pmax_rolling_std_7"] = grouped["p_max"].apply(lambda s: s.rolling(7, min_periods=2).std()).reset_index(level=[0, 1], drop=True).fillna(0)

    scored["near_set_flag"] = scored["ratio"] >= 0.95
    scored["over_set_flag"] = scored["ratio"] >= 1.00
    scored["rise_flag"] = scored["recent_jump"] > 0
    scored["activity_flag"] = scored["Activity"] > 0

    scored["continuous_rise_days"] = grouped["rise_flag"].apply(_consecutive_counts).reset_index(level=[0, 1], drop=True)
    scored["near_set_days"] = grouped["near_set_flag"].apply(_consecutive_counts).reset_index(level=[0, 1], drop=True)
    scored["activity_days"] = grouped["activity_flag"].apply(_consecutive_counts).reset_index(level=[0, 1], drop=True)
    return scored


def compute_mechanistic_health(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    scored = extract_domain_features(df)
    hi = np.full(len(scored), 100.0)

    hi -= np.where(scored["ratio"] >= 1.00, 35, 0)
    hi -= np.where((scored["ratio"] >= 0.98) & (scored["ratio"] < 1.00), 20, 0)
    hi -= np.where((scored["ratio"] >= 0.95) & (scored["ratio"] < 0.98), 10, 0)
    hi -= np.where(scored["slope_3d"] > 0.01, 10, 0)
    hi -= np.where(scored["slope_3d"] > 0.02, 10, 0)
    hi -= np.where(scored["continuous_rise_days"] >= 2, 6, 0)
    hi -= np.where(scored["near_set_days"] >= 2, 8, 0)
    hi -= np.where(scored["volatility_3d"] > 0.015, 6, 0)
    hi -= np.where(scored["recent_jump"] > 0.02, 8, 0)

    hi -= scored.get("psv_act", 0).fillna(0) * 30
    hi -= scored.get("psv_weeping", 0).fillna(0) * 15

    hi -= np.where(
        (scored.get("temp", 0).fillna(0) >= 33)
        & (scored.get("level", 0).fillna(0) >= 80)
        & (scored["ratio"] >= 0.95),
        10,
        0,
    )

    scored["HI"] = np.clip(hi, 0, 100)
    scored["Risk"] = scored["HI"].apply(risk_from_hi)
    return scored
