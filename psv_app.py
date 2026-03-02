

import os
import uuid
from datetime import timedelta
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib as mpl
from matplotlib import font_manager


# ================== Config ==================
SET_P = 1.32
DATA_FILE = "psv_data.csv"
ALERT_FILE = "psv_alerts.csv"
AUDIT_FILE = "psv_audit_logs.csv"

TABLE_DATA = "psv_data"
TABLE_ALERT = "psv_alerts"
TABLE_AUDIT = "psv_audit_logs"

STATUS_FLOW = ["待确认", "已派工", "处理中", "已验证", "已关闭"]
STATIONS = ["华盘LNG加气站", "罗所LNG加气站"]
DEFAULT_STATION = "华盘LNG加气站"

AI_FEATURES = ["p_now", "p_max", "ratio", "slope_3d", "level", "temp", "Activity"]
AI_FEATURE_LABELS = {
    "p_now": "当前压力",
    "p_max": "最高压力",
    "ratio": "接近整定比",
    "slope_3d": "3日压力斜率",
    "level": "液位",
    "temp": "温度",
    "Activity": "动作/微放散",
}


# ================== Supabase init ==================
try:
    from supabase import create_client

    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False


def _secret_get(key: str, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


SUPABASE_URL = _secret_get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = _secret_get("SUPABASE_KEY", "") or os.getenv("SUPABASE_KEY", "")

USE_SUPABASE = True
supabase = None
if USE_SUPABASE and SUPABASE_OK and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
elif USE_SUPABASE:
    st.sidebar.warning("⚠️ 未检测到 Supabase 配置（SUPABASE_URL / SUPABASE_KEY），将回退为本地CSV存储。")
    USE_SUPABASE = False


# ================== ML ==================
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ================== Font ==================
def _setup_cjk_font():
    preferred_font_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_font_names:
        if name in available:
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    if os.path.exists(font_path):
        fp = font_manager.FontProperties(fname=font_path)
        mpl.rcParams["font.family"] = fp.get_name()
        mpl.rcParams["axes.unicode_minus"] = False


_setup_cjk_font()
st.set_page_config(page_title="Isolation Forest LNG安全阀AI预警系统", layout="wide")


# ================== Auth ==================
def load_accounts() -> dict:
    hp = _secret_get("PASSWORD_HUAPAN", "hp123456") or os.getenv("PASSWORD_HUAPAN", "hp123456")
    ls = _secret_get("PASSWORD_LUOSUO", "ls123456") or os.getenv("PASSWORD_LUOSUO", "ls123456")
    leader = _secret_get("PASSWORD_LEADER", "leader123456") or os.getenv("PASSWORD_LEADER", "leader123456")

    return {
        "华盘站": {"password": hp, "role": "station", "station_scope": "华盘LNG加气站"},
        "罗所站": {"password": ls, "role": "station", "station_scope": "罗所LNG加气站"},
        "领导": {"password": leader, "role": "leader", "station_scope": "ALL"},
    }


ACCOUNTS = load_accounts()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "role" not in st.session_state:
    st.session_state.role = ""
if "station_scope" not in st.session_state:
    st.session_state.station_scope = ""

st.sidebar.title("🔐 访问控制")

if not st.session_state.authenticated:
    login_name = st.sidebar.selectbox("账号", list(ACCOUNTS.keys()), index=0)
    login_pwd = st.sidebar.text_input("密码", type="password")
    if st.sidebar.button("登录", use_container_width=True):
        if login_pwd == ACCOUNTS[login_name]["password"]:
            st.session_state.authenticated = True
            st.session_state.user_name = login_name
            st.session_state.role = ACCOUNTS[login_name]["role"]
            st.session_state.station_scope = ACCOUNTS[login_name]["station_scope"]
            st.rerun()
        else:
            st.sidebar.error("密码错误")
    st.warning("请输入账号和密码后进入系统。")
    st.stop()
else:
    st.sidebar.success(
        f"已登录：{st.session_state.user_name} | 角色：{st.session_state.role} | 范围：{st.session_state.station_scope}"
    )
    if st.sidebar.button("退出登录", use_container_width=True):
        for k in ["authenticated", "user_name", "role", "station_scope"]:
            st.session_state.pop(k, None)
        st.rerun()


ROLE = st.session_state.role
STATION_SCOPE = st.session_state.station_scope
IS_LEADER = ROLE == "leader"


# ================== Data helpers ==================
BASE_DATA_COLS = [
    "date",
    "station",
    "valve_type",
    "p_now",
    "p_max",
    "level",
    "temp",
    "psv_act",
    "psv_weeping",
    "operator_role",
    "operator_name",
    "updated_at",
]

BASE_ALERT_COLS = [
    "id",
    "date",
    "station",
    "valve_type",
    "risk_level",
    "trigger_source",
    "trigger_detail",
    "status",
    "owner",
    "action_taken",
    "verification_result",
    "created_at",
    "updated_at",
    "closed_at",
]

BASE_AUDIT_COLS = ["id", "entity_type", "entity_id", "action", "operator", "payload", "created_at"]


def _ensure_local_files():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=BASE_DATA_COLS).to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    if not os.path.exists(ALERT_FILE):
        pd.DataFrame(columns=BASE_ALERT_COLS).to_csv(ALERT_FILE, index=False, encoding="utf-8-sig")
    if not os.path.exists(AUDIT_FILE):
        pd.DataFrame(columns=BASE_AUDIT_COLS).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")


_ensure_local_files()


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df

    rename_map = {
        "Valve_type": "valve_type",
        "P_now": "p_now",
        "P_max": "p_max",
        "Level": "level",
        "Temp": "temp",
        "PSV_act": "psv_act",
        "PSV_weeping": "psv_weeping",
    }
    df = df.rename(columns=rename_map)

    for c in BASE_DATA_COLS:
        if c not in df.columns:
            if c == "station":
                df[c] = DEFAULT_STATION
            elif c in ["operator_role", "operator_name", "updated_at"]:
                df[c] = ""
            else:
                df[c] = np.nan

    return df[BASE_DATA_COLS]


def _normalize_df(df0: pd.DataFrame) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return pd.DataFrame(columns=BASE_DATA_COLS)

    df = _standardize_columns(df0.copy())
    df["station"] = df["station"].fillna(DEFAULT_STATION).replace("", DEFAULT_STATION)
    df["station"] = df["station"].where(df["station"].isin(STATIONS), DEFAULT_STATION)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    for col in ["p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["p_now"] = df["p_now"].clip(lower=0, upper=2)
    df["p_max"] = df["p_max"].clip(lower=0, upper=2)
    df["level"] = df["level"].clip(lower=0, upper=100)
    df["temp"] = df["temp"].clip(lower=-50, upper=80)

    m = df["p_max"].notna() & df["p_now"].notna() & (df["p_max"] < df["p_now"])
    if m.any():
        df.loc[m, "p_max"] = df.loc[m, "p_now"]

    df = df.dropna(subset=["date", "station", "valve_type", "p_max"])
    df = (
        df.sort_values(["station", "valve_type", "date", "updated_at"])
        .drop_duplicates(subset=["date", "station", "valve_type"], keep="last")
        .reset_index(drop=True)
    )

    return df


def _scope_filter(df: pd.DataFrame, station_scope: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    if station_scope == "ALL":
        return df
    return df[df["station"] == station_scope].copy()


def load_data(station_scope: str, role: str) -> pd.DataFrame:
    if USE_SUPABASE and supabase is not None:
        resp = supabase.table(TABLE_DATA).select("*").execute()
        raw = pd.DataFrame(resp.data or [])
    else:
        raw = pd.read_csv(DATA_FILE)

    df = _normalize_df(raw)
    return _scope_filter(df, station_scope)


def _write_local_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def save_record(record: dict, station_scope: str, role: str) -> None:
    if role == "leader":
        raise PermissionError("领导账号为只读，不允许写入数据")

    if station_scope != "ALL" and record.get("station") != station_scope:
        raise PermissionError("只能写入本站数据")

    record = record.copy()
    record["updated_at"] = pd.Timestamp.now().isoformat()

    if USE_SUPABASE and supabase is not None:
        supabase.table(TABLE_DATA).upsert(record, on_conflict="date,station,valve_type").execute()
        return

    df_all = _normalize_df(pd.read_csv(DATA_FILE))
    new_row = pd.DataFrame([record])
    merged = pd.concat([df_all, new_row], ignore_index=True)
    merged = _normalize_df(merged)
    _write_local_data(merged)

# ================== Alerts ==================
def _normalize_alert_df(df0: pd.DataFrame) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return pd.DataFrame(columns=BASE_ALERT_COLS)

    df = df0.copy()
    for c in BASE_ALERT_COLS:
        if c not in df.columns:
            df[c] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["station"] = df["station"].fillna(DEFAULT_STATION).replace("", DEFAULT_STATION)
    df["status"] = df["status"].replace("", "待确认")
    return df[BASE_ALERT_COLS]


def _load_alerts_all() -> pd.DataFrame:
    if USE_SUPABASE and supabase is not None:
        resp = supabase.table(TABLE_ALERT).select("*").execute()
        raw = pd.DataFrame(resp.data or [])
    else:
        raw = pd.read_csv(ALERT_FILE)
    return _normalize_alert_df(raw)


def _save_alerts_local(df: pd.DataFrame):
    df.to_csv(ALERT_FILE, index=False, encoding="utf-8-sig")


def _safe_num(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


def _normalize_trigger_detail(v):
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if isinstance(x, (int, float, np.number)):
                out[k] = _safe_num(x)
            else:
                out[k] = x
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return {}
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
            return {"raw": obj}
        except Exception:
            return {"raw": s}
    return {}


def append_audit(entity_type: str, entity_id: str, action: str, operator: str, payload: str):
    log = {
        "id": str(uuid.uuid4()),
        "entity_type": entity_type,
        "entity_id": str(entity_id),
        "action": action,
        "operator": operator,
        "payload": payload,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    if USE_SUPABASE and supabase is not None:
        supabase.table(TABLE_AUDIT).insert(log).execute()
    else:
        df = pd.read_csv(AUDIT_FILE)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
        df.to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")


def _find_alert(alerts: pd.DataFrame, date_value, station: str, valve_type: str):
    m = (
        (alerts["date"] == pd.to_datetime(date_value).date())
        & (alerts["station"] == station)
        & (alerts["valve_type"] == valve_type)
    )
    return alerts[m]


def create_or_update_alert(record: dict):
    now_iso = pd.Timestamp.now().isoformat()
    alerts = _load_alerts_all()

    found = _find_alert(alerts, record["date"], record["station"], record["valve_type"])
    if len(found) > 0:
        idx = found.index[0]
        keep_status = alerts.loc[idx, "status"] or "待确认"
        detail_obj = _normalize_trigger_detail(record.get("trigger_detail", alerts.loc[idx, "trigger_detail"]))
        alerts.loc[idx, "risk_level"] = record.get("risk_level", alerts.loc[idx, "risk_level"])
        alerts.loc[idx, "trigger_source"] = record.get("trigger_source", alerts.loc[idx, "trigger_source"])
        # Store JSON text in dataframe to avoid pandas aligning dict keys as a Series.
        alerts.at[idx, "trigger_detail"] = json.dumps(detail_obj, ensure_ascii=False)
        alerts.loc[idx, "updated_at"] = now_iso
        alerts.loc[idx, "status"] = keep_status
    else:
        detail_obj = _normalize_trigger_detail(record.get("trigger_detail", {}))
        new_alert = {
            "id": str(uuid.uuid4()),
            "date": str(pd.to_datetime(record["date"]).date()),
            "station": record["station"],
            "valve_type": record["valve_type"],
            "risk_level": record.get("risk_level", "🔴 高风险"),
            "trigger_source": record.get("trigger_source", "rule"),
            "trigger_detail": json.dumps(detail_obj, ensure_ascii=False),
            "status": "待确认",
            "owner": "",
            "action_taken": "",
            "verification_result": "",
            "created_at": now_iso,
            "updated_at": now_iso,
            "closed_at": "",
        }
        alerts = pd.concat([alerts, pd.DataFrame([new_alert])], ignore_index=True)

    if USE_SUPABASE and supabase is not None:
        record_date = str(pd.to_datetime(record["date"]).date())
        base_payload = {
            "date": record_date,
            "station": record["station"],
            "valve_type": record["valve_type"],
            "risk_level": record.get("risk_level", "🔴 高风险"),
            "trigger_source": record.get("trigger_source", "rule"),
            "trigger_detail": _normalize_trigger_detail(record.get("trigger_detail", {})),
            "updated_at": now_iso,
        }

        q = (
            supabase.table(TABLE_ALERT)
            .select("id,status,owner,action_taken,verification_result,created_at,closed_at")
            .eq("date", record_date)
            .eq("station", record["station"])
            .eq("valve_type", record["valve_type"])
            .limit(1)
            .execute()
        )
        ex = (q.data or [])
        if ex:
            old = ex[0]
            payload = {
                **base_payload,
                "id": old.get("id"),
                "status": old.get("status") or "待确认",
                "owner": old.get("owner") or "",
                "action_taken": old.get("action_taken") or "",
                "verification_result": old.get("verification_result") or "",
                "created_at": old.get("created_at") or now_iso,
                "closed_at": old.get("closed_at"),
            }
        else:
            payload = {
                **base_payload,
                "id": str(uuid.uuid4()),
                "status": "待确认",
                "owner": "",
                "action_taken": "",
                "verification_result": "",
                "created_at": now_iso,
                "closed_at": None,
            }
        supabase.table(TABLE_ALERT).upsert(payload, on_conflict="date,station,valve_type").execute()
    else:
        _save_alerts_local(alerts)


def update_alert_status(alert_id: str, new_status: str, operator: str, action_taken: str = "", verification_result: str = ""):
    alerts = _load_alerts_all()
    hit = alerts[alerts["id"].astype(str) == str(alert_id)]
    if len(hit) == 0:
        raise ValueError("未找到告警")

    idx = hit.index[0]
    cur = alerts.loc[idx, "status"]
    if cur not in STATUS_FLOW:
        cur = "待确认"
    if new_status not in STATUS_FLOW:
        raise ValueError("非法状态")

    cur_i = STATUS_FLOW.index(cur)
    new_i = STATUS_FLOW.index(new_status)
    if not (new_i == cur_i or new_i == cur_i + 1):
        raise ValueError("状态仅允许保持不变或推进一步")

    if new_status == "已关闭":
        if not action_taken.strip() or not verification_result.strip():
            raise ValueError("关闭告警前必须填写整改措施和复验结果")
        alerts.loc[idx, "closed_at"] = pd.Timestamp.now().isoformat()

    if action_taken.strip():
        alerts.loc[idx, "action_taken"] = action_taken.strip()
    if verification_result.strip():
        alerts.loc[idx, "verification_result"] = verification_result.strip()

    alerts.loc[idx, "status"] = new_status
    alerts.loc[idx, "owner"] = operator
    alerts.loc[idx, "updated_at"] = pd.Timestamp.now().isoformat()

    if USE_SUPABASE and supabase is not None:
        updates = {
            "status": new_status,
            "owner": operator,
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        if action_taken.strip():
            updates["action_taken"] = action_taken.strip()
        if verification_result.strip():
            updates["verification_result"] = verification_result.strip()
        if new_status == "已关闭":
            updates["closed_at"] = pd.Timestamp.now().isoformat()
        supabase.table(TABLE_ALERT).update(updates).eq("id", str(alert_id)).execute()
    else:
        _save_alerts_local(alerts)

    append_audit(
        entity_type="alert",
        entity_id=str(alert_id),
        action=f"status:{cur}->{new_status}",
        operator=operator,
        payload=f"action_taken={action_taken}; verification_result={verification_result}",
    )


def list_alerts(station_scope: str, role: str) -> pd.DataFrame:
    alerts = _load_alerts_all()
    if station_scope != "ALL":
        alerts = alerts[alerts["station"] == station_scope].copy()
    if len(alerts) == 0:
        return alerts
    return alerts.sort_values(["status", "date"], ascending=[True, False]).reset_index(drop=True)


# ================== Scoring ==================
def _risk_from_hi(x: float) -> str:
    if x >= 85:
        return "🟢 安全"
    if x >= 70:
        return "🟡 预警"
    return "🔴 高风险"


def build_ai_reason_top_features(row: pd.Series, group_stats: Dict[str, pd.Series]) -> Tuple[str, str, str]:
    med = group_stats.get("median", pd.Series(dtype=float))
    std = group_stats.get("std", pd.Series(dtype=float))

    scores = {}
    for f in AI_FEATURES:
        v = pd.to_numeric(pd.Series([row.get(f, np.nan)]), errors="coerce").iloc[0]
        m = pd.to_numeric(pd.Series([med.get(f, np.nan)]), errors="coerce").iloc[0]
        s = pd.to_numeric(pd.Series([std.get(f, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(v) or pd.isna(m):
            scores[f] = -np.inf
            continue
        if pd.isna(s) or float(s) == 0:
            s = 1.0
        scores[f] = abs((float(v) - float(m)) / float(s))

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    labels = [f"{AI_FEATURE_LABELS.get(k, k)}偏离" if np.isfinite(v) else "-" for k, v in top]
    while len(labels) < 3:
        labels.append("-")
    return labels[0], labels[1], labels[2]


def compute_iforest_signals(df_grouped: pd.DataFrame, contamination: float, window_days: int = 60, min_samples: int = 30) -> pd.DataFrame:
    df = df_grouped.copy()
    df["ai_raw_score"] = np.nan
    df["ai_score_pct"] = np.nan
    df["ai_observe_flag"] = False
    df["ai_escalate_flag"] = False
    df["ai_reason_top1"] = "-"
    df["ai_reason_top2"] = "-"
    df["ai_reason_top3"] = "-"

    if len(df) == 0 or not SKLEARN_OK:
        df["AI_anomaly"] = False
        df["AI_score"] = np.nan
        return df

    for _, g in df.groupby(["station", "valve_type"], sort=False):
        g = g.sort_values("date")
        ordered_idx = list(g.index)
        stats_cache = {}

        for pos, idx in enumerate(ordered_idx):
            win_idx = ordered_idx[max(0, pos - (window_days - 1)) : pos + 1]
            xw = g.loc[win_idx, AI_FEATURES].copy()
            xw = xw.apply(pd.to_numeric, errors="coerce")
            med = xw.median(numeric_only=True)
            xw = xw.fillna(med).fillna(0.0)

            if len(xw) < min_samples:
                continue

            std = xw.std(numeric_only=True).replace(0, np.nan)

            scaler = StandardScaler()
            xs = scaler.fit_transform(xw.values)

            iso = IsolationForest(
                n_estimators=300,
                contamination=float(contamination),
                random_state=42,
            )
            iso.fit(xs)

            cur_x = xs[-1].reshape(1, -1)
            raw = float(-iso.score_samples(cur_x)[0])
            df.at[idx, "ai_raw_score"] = raw
            stats_cache[idx] = {"median": med, "std": std}

        valid_idx = [i for i in ordered_idx if pd.notna(df.at[i, "ai_raw_score"])]
        if len(valid_idx) == 0:
            continue

        raw_s = pd.to_numeric(df.loc[valid_idx, "ai_raw_score"], errors="coerce")
        pct = raw_s.rank(pct=True, method="average") * 100
        df.loc[valid_idx, "ai_score_pct"] = pct
        df.loc[valid_idx, "ai_observe_flag"] = df.loc[valid_idx, "ai_score_pct"] >= 95

        for pos, idx in enumerate(ordered_idx):
            if idx not in valid_idx:
                continue

            observe = bool(df.at[idx, "ai_observe_flag"])
            if not observe:
                continue

            cur_date = pd.to_datetime(df.at[idx, "date"], errors="coerce")
            rule_risky = str(df.at[idx, "Risk"]) in ["🟡 预警", "🔴 高风险"]

            prev_consecutive = False
            if pos > 0:
                prev_idx = ordered_idx[pos - 1]
                prev_observe = bool(df.at[prev_idx, "ai_observe_flag"])
                prev_date = pd.to_datetime(df.at[prev_idx, "date"], errors="coerce")
                if pd.notna(cur_date) and pd.notna(prev_date):
                    prev_consecutive = prev_observe and ((cur_date.date() - prev_date.date()).days == 1)

            escalate = prev_consecutive or rule_risky
            df.at[idx, "ai_escalate_flag"] = bool(escalate)

            t1, t2, t3 = build_ai_reason_top_features(df.loc[idx], stats_cache.get(idx, {}))
            df.at[idx, "ai_reason_top1"] = t1
            df.at[idx, "ai_reason_top2"] = t2
            df.at[idx, "ai_reason_top3"] = t3

    df["AI_anomaly"] = df["ai_observe_flag"].fillna(False).astype(bool)
    df["AI_score"] = df["ai_raw_score"]
    return df


def compute_scores(df0: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return df0

    df = _normalize_df(df0).copy()
    df = df.sort_values(["station", "valve_type", "date"]).reset_index(drop=True)

    df["ratio"] = df["p_max"] / SET_P
    df["slope_3d"] = (
        df.groupby(["station", "valve_type"])["p_max"]
        .apply(lambda s: s.diff().rolling(3).mean())
        .reset_index(level=[0, 1], drop=True)
    )
    df["slope"] = df["slope_3d"]

    hi = np.full(len(df), 100.0)
    hi -= np.where(df["ratio"] >= 1.00, 35, 0)
    hi -= np.where((df["ratio"] >= 0.98) & (df["ratio"] < 1.00), 20, 0)
    hi -= np.where((df["ratio"] >= 0.95) & (df["ratio"] < 0.98), 10, 0)

    hi -= np.where(df["slope_3d"] > 0.01, 10, 0)
    hi -= np.where(df["slope_3d"] > 0.02, 10, 0)

    hi -= df.get("psv_act", 0).fillna(0) * 30
    hi -= df.get("psv_weeping", 0).fillna(0) * 15

    hi -= np.where(
        (df.get("temp", 0).fillna(0) >= 33)
        & (df.get("level", 0).fillna(0) >= 80)
        & (df["ratio"] >= 0.95),
        10,
        0,
    )

    df["HI"] = np.clip(hi, 0, 100)
    df["Risk"] = df["HI"].apply(_risk_from_hi)
    df["Activity"] = df.get("psv_act", 0).fillna(0) + df.get("psv_weeping", 0).fillna(0)

    df["ai_raw_score"] = np.nan
    df["ai_score_pct"] = np.nan
    df["ai_observe_flag"] = False
    df["ai_escalate_flag"] = False
    df["ai_reason_top1"] = "-"
    df["ai_reason_top2"] = "-"
    df["ai_reason_top3"] = "-"
    df["AI_anomaly"] = False
    df["AI_score"] = np.nan

    if enable_ai and SKLEARN_OK:
        df = compute_iforest_signals(df, contamination=contamination, window_days=60, min_samples=30)
        obs = df["ai_observe_flag"].fillna(False).astype(bool)
        esc = df["ai_escalate_flag"].fillna(False).astype(bool)

        penalty = obs.astype(int) * 6 + esc.astype(int) * 10
        df["HI_final"] = np.clip(df["HI"] - penalty, 0, 100)
        df["Risk_final"] = df["HI_final"].apply(_risk_from_hi)
        df.loc[obs & (df["Risk_final"] == "🟢 安全"), "Risk_final"] = "🟡 预警"
        df.loc[esc, "Risk_final"] = "🔴 高风险"
    else:
        df["HI_final"] = df["HI"]
        df["Risk_final"] = df["Risk"]

    return df


def _calc_trigger_source(row: pd.Series):
    rule_hit = str(row.get("Risk_final", "")) == "🔴 高风险"
    ai_hit = bool(row.get("ai_escalate_flag", False))

    if rule_hit and ai_hit:
        return "both"
    if rule_hit:
        return "rule"
    if ai_hit:
        return "ai"
    return ""


def sync_alerts_from_scores(df_scored: pd.DataFrame):
    if df_scored is None or len(df_scored) == 0:
        return

    for _, row in df_scored.iterrows():
        source = _calc_trigger_source(row)
        if not source:
            continue

        create_or_update_alert(
            {
                "date": row["date"],
                "station": row["station"],
                "valve_type": row["valve_type"],
                "risk_level": row.get("Risk_final", "🔴 高风险"),
                "trigger_source": source,
                "trigger_detail": {
                    "HI_final": float(row.get("HI_final", np.nan)),
                    "ratio": float(row.get("ratio", np.nan)),
                    "ai_score_pct": float(row.get("ai_score_pct", np.nan)) if pd.notna(row.get("ai_score_pct", np.nan)) else None,
                    "ai_reason_top1": row.get("ai_reason_top1", "-"),
                    "ai_reason_top2": row.get("ai_reason_top2", "-"),
                    "ai_reason_top3": row.get("ai_reason_top3", "-"),
                },
            }
        )

# ================== UI ==================
st.title("基于Isolation Forest算法的LNG储罐安全阀健康监测与风险AI预警系统")
st.caption("双站点分角色管理｜滚动60天Isolation Forest｜双条件AI升级预警")

st.sidebar.divider()
st.sidebar.header("🧠 AI 参数")
enable_ai = st.sidebar.checkbox("启用 Isolation Forest", value=True)
contamination = st.sidebar.slider("模型异常比例（越大越敏感）", min_value=0.02, max_value=0.20, value=0.08, step=0.01)
if enable_ai and not SKLEARN_OK:
    st.sidebar.warning("当前环境缺少 scikit-learn，AI异常检测不可用。")
    enable_ai = False

if not IS_LEADER:
    st.sidebar.divider()
    st.sidebar.header("📝 本站数据录入")
    st.sidebar.info(f"当前站点：{STATION_SCOPE}")

    valve_type = st.sidebar.selectbox("选择安全阀类型", ["泵后安全阀", "储罐主阀", "储罐辅阀"])
    date = st.sidebar.date_input("日期")
    p_now = st.sidebar.number_input("当前压力 p_now (MPa)", 0.0, 2.0, 1.20, 0.01)
    p_max = st.sidebar.number_input(
        "当日最高压力 p_max (MPa)",
        0.0,
        2.0,
        1.20,
        0.01,
        help="建议：p_max ≥ p_now；若输入小于 p_now，系统会自动按 p_now 修正。",
    )
    level = st.sidebar.number_input("液位 level (%)", 0, 100, 60)
    temp = st.sidebar.number_input("环境温度 temp (℃)", -30, 60, 25)
    psv_act = st.sidebar.selectbox("是否动作", ["否", "是"])
    psv_weeping = st.sidebar.selectbox("是否微放散/嘶嘶声", ["否", "是"])

    if st.sidebar.button("保存并计算", use_container_width=True):
        p_now_f = float(p_now)
        p_max_f = float(p_max)
        if p_max_f < p_now_f:
            st.sidebar.warning(f"已自动修正：p_max({p_max_f:.2f}) < p_now({p_now_f:.2f})，将 p_max 设为 {p_now_f:.2f}")
            p_max_f = p_now_f

        try:
            save_record(
                {
                    "date": str(date),
                    "station": STATION_SCOPE,
                    "valve_type": valve_type,
                    "p_now": p_now_f,
                    "p_max": p_max_f,
                    "level": int(level),
                    "temp": int(temp),
                    "psv_act": 1 if psv_act == "是" else 0,
                    "psv_weeping": 1 if psv_weeping == "是" else 0,
                    "operator_role": ROLE,
                    "operator_name": st.session_state.user_name,
                },
                station_scope=STATION_SCOPE,
                role=ROLE,
            )
            st.sidebar.success("✅ 数据已保存")
            st.rerun()
        except Exception as ex:
            st.sidebar.error(f"保存失败：{ex}")
else:
    st.sidebar.info("领导账号为只读模式，不可录入或修改原始数据。")


# Load + score + auto alerts
df_raw = load_data(station_scope=STATION_SCOPE, role=ROLE)
df = compute_scores(df_raw, enable_ai=enable_ai, contamination=contamination)
sync_alerts_from_scores(df)
alerts = list_alerts(station_scope=STATION_SCOPE, role=ROLE)

if len(df) == 0:
    st.info("当前权限范围内还没有数据。")
    st.stop()


# Global date filter (default 30 days)
min_d, max_d = df["date"].min(), df["date"].max()
default_start = max(min_d, max_d - timedelta(days=29))
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    start_date = st.date_input("开始日期", value=default_start, min_value=min_d, max_value=max_d, key="global_start")
with c2:
    end_date = st.date_input("结束日期", value=max_d, min_value=min_d, max_value=max_d, key="global_end")
with c3:
    st.caption("默认窗口：近30天。页面采用分Tab结构，避免长滚动。")

df_f = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
alerts_f = alerts[
    (pd.to_datetime(alerts["date"], errors="coerce").dt.date >= start_date)
    & (pd.to_datetime(alerts["date"], errors="coerce").dt.date <= end_date)
].copy()

if len(df_f) == 0:
    st.warning("所选日期范围内无数据。")
    st.stop()


def _slice_by_station(df_input: pd.DataFrame, station_pick: str) -> pd.DataFrame:
    if len(df_input) == 0:
        return df_input
    if station_pick == "全部站点":
        return df_input.copy()
    return df_input[df_input["station"] == station_pick].copy()


def build_hi_heatmap(df_filtered: pd.DataFrame) -> pd.DataFrame:
    if len(df_filtered) == 0:
        return pd.DataFrame()
    heat = df_filtered.pivot_table(index="valve_type", columns="date", values="HI_final", aggfunc="mean")
    return heat.sort_index()


def build_hi_compare(df_filtered: pd.DataFrame) -> pd.DataFrame:
    if len(df_filtered) == 0:
        return pd.DataFrame()
    return (
        df_filtered.groupby("valve_type")
        .agg(
            avg_HI=("HI_final", "mean"),
            min_HI=("HI_final", "min"),
            red_days=("Risk_final", lambda s: (s == "🔴 高风险").sum()),
            yellow_days=("Risk_final", lambda s: (s == "🟡 预警").sum()),
        )
        .reset_index()
        .sort_values("avg_HI")
    )


def build_pressure_trend(df_filtered: pd.DataFrame, station: str, valve: str) -> pd.DataFrame:
    sdf = _slice_by_station(df_filtered, station)
    if len(sdf) == 0:
        return sdf
    vdf = sdf[sdf["valve_type"] == valve].copy()
    if len(vdf) == 0:
        return vdf
    if station == "全部站点":
        vdf = (
            vdf.groupby("date", as_index=False)
            .agg(
                p_now=("p_now", "mean"),
                p_max=("p_max", "mean"),
                ai_observe_flag=("ai_observe_flag", "max"),
            )
            .sort_values("date")
        )
    else:
        vdf = vdf.sort_values("date")
    vdf["date_dt"] = pd.to_datetime(vdf["date"])
    return vdf


def build_leader_storyline(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame) -> str:
    if len(df_filtered) == 0:
        return "当前范围暂无数据。"

    comp = build_hi_compare(df_filtered)
    worst_name = "—"
    worst_hi = np.nan
    if len(comp) > 0:
        worst = comp.iloc[0]
        worst_name = str(worst["valve_type"])
        worst_hi = float(worst["avg_HI"])

    last_day = df_filtered["date"].max()
    recent7 = df_filtered[df_filtered["date"] >= (last_day - timedelta(days=6))]["HI_final"].mean()
    prev7 = df_filtered[
        (df_filtered["date"] >= (last_day - timedelta(days=13)))
        & (df_filtered["date"] <= (last_day - timedelta(days=7)))
    ]["HI_final"].mean()

    if np.isnan(recent7) or np.isnan(prev7):
        trend_text = "趋势样本不足"
    else:
        delta = recent7 - prev7
        trend_text = f"近7天较前7天 {'上升' if delta >= 0 else '下降'} {abs(delta):.1f}"

    ai_obs = int(df_filtered["ai_observe_flag"].sum()) if "ai_observe_flag" in df_filtered.columns else 0
    ai_esc = int(df_filtered["ai_escalate_flag"].sum()) if "ai_escalate_flag" in df_filtered.columns else 0

    close_rate = 0.0
    if len(alerts_filtered) > 0:
        close_rate = float((alerts_filtered["status"] == "已关闭").mean() * 100)

    return (
        f"重点阀门：{worst_name}（平均HI {worst_hi:.1f}）｜{trend_text}｜"
        f"AI观察异常 {ai_obs} 次，AI升级预警 {ai_esc} 次｜告警闭环率 {close_rate:.1f}%"
    )

def render_tab_history(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame):
    st.subheader("📈 历史分析")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="hist_station")
    else:
        station_pick = STATION_SCOPE
        st.info(f"当前站点：{station_pick}")

    hist_df = _slice_by_station(df_filtered, station_pick)
    hist_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(hist_df) == 0:
        st.warning("该范围内暂无数据。")
        return

    valve_opts = sorted(hist_df["valve_type"].unique())
    valve_pick = st.selectbox("阀门", valve_opts, key="hist_valve")
    vdf = build_pressure_trend(hist_df, "全部站点" if station_pick == "全部站点" else station_pick, valve_pick)

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        st.markdown("**3线压力趋势图**")
        if len(vdf) == 0:
            st.info("暂无可绘制数据")
        else:
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            ax.plot(vdf["date_dt"], vdf["p_now"], marker="o", linestyle="--", label="p_now")
            ax.plot(vdf["date_dt"], vdf["p_max"], marker="o", label="p_max")
            ax.axhline(SET_P, linestyle="-.", color="#6d4c41", label=f"整定线 {SET_P:.2f}MPa")

            ai_points = vdf[vdf["ai_observe_flag"] == True]
            if len(ai_points) > 0:
                ax.scatter(ai_points["date_dt"], ai_points["p_max"], color="#d32f2f", zorder=4, label="AI异常点")

            ax.set_ylabel("MPa")
            ax.set_title("Pressure Trend (3 lines)")
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            plt.xticks(rotation=30)
            ax.legend(fontsize=8)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            pmax_start = float(vdf["p_max"].iloc[0])
            pmax_end = float(vdf["p_max"].iloc[-1])
            pmax_delta = pmax_end - pmax_start
            near_set_days = int((vdf["p_max"] >= SET_P * 0.95).sum())
            exceed_set_days = int((vdf["p_max"] >= SET_P).sum())
            ai_days = int(vdf["ai_observe_flag"].sum()) if "ai_observe_flag" in vdf.columns else 0
            if pmax_delta > 0.02:
                trend_word = "明显上升"
            elif pmax_delta < -0.02:
                trend_word = "下降"
            else:
                trend_word = "总体平稳"

            st.caption(
                "趋势解读："
                f"p_max 从 {pmax_start:.2f}MPa 变化到 {pmax_end:.2f}MPa（{trend_word}，变化 {pmax_delta:+.2f}MPa）。"
                f"近整定压力(≥{SET_P*0.95:.2f}MPa)共 {near_set_days} 天，超过整定线共 {exceed_set_days} 天，"
                f"AI观察异常 {ai_days} 天。"
            )
            if exceed_set_days > 0:
                st.warning("建议：出现超过整定线的日期应优先复盘工况与阀门动作记录。")
            elif near_set_days > 0:
                st.info("建议：压力已多次接近整定线，建议提前做维护巡检，避免突发动作。")

    with c2:
        st.markdown("**HI热力图**")
        heat = build_hi_heatmap(hist_df)
        if len(heat) == 0:
            st.info("暂无可绘制数据")
        else:
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(list(heat.index))

            cols = list(heat.columns)
            if len(cols) <= 8:
                tick_idx = list(range(len(cols)))
            else:
                tick_idx = sorted(set(np.linspace(0, len(cols) - 1, 7).round().astype(int).tolist()))
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([pd.to_datetime(cols[i]).strftime("%m-%d") for i in tick_idx], rotation=30, ha="right")

            ax.set_title("HI Heatmap")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            flat = heat.stack(dropna=True)
            if len(flat) > 0:
                worst_idx = flat.idxmin()
                worst_hi = float(flat.min())
                worst_valve = str(worst_idx[0])
                worst_date = pd.to_datetime(worst_idx[1]).strftime("%Y-%m-%d")
            else:
                worst_hi = np.nan
                worst_valve = "-"
                worst_date = "-"

            yellow_cnt = int((hist_df["Risk_final"] == "🟡 预警").sum())
            red_cnt = int((hist_df["Risk_final"] == "🔴 高风险").sum())
            st.caption(
                "热力图解读：颜色越偏红代表HI越低、风险越高；越偏绿代表运行更稳定。"
                f"本周期最低HI为 {worst_hi:.1f}（{worst_date}，{worst_valve}），"
                f"累计预警 {yellow_cnt} 条，高风险 {red_cnt} 条。"
            )
            if red_cnt > 0:
                st.warning("建议：优先处理热力图中“连续偏黄/偏红”的阀门与日期段。")

    with c3:
        st.markdown("**阀门HI对比**")
        comp = build_hi_compare(hist_df)
        if len(comp) == 0:
            st.info("暂无可绘制数据")
        else:
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            ax.bar(comp["valve_type"], comp["avg_HI"], color="#2e7d32")
            ax.set_ylim(0, 100)
            ax.set_ylabel("avg HI")
            ax.set_title("Valve HI Compare")
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            st.dataframe(comp[["valve_type", "avg_HI", "min_HI", "red_days", "yellow_days"]], use_container_width=True, height=170)

    st.info(build_leader_storyline(hist_df, hist_alerts))


def render_tab_ai(df_filtered: pd.DataFrame):
    st.subheader("🤖 AI预警中心（Isolation Forest）")
    st.caption("模型口径：按 station+valve_type 分组、滚动60天训练、最小样本30、双条件升级")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="ai_station")
    else:
        station_pick = STATION_SCOPE

    ai_df = _slice_by_station(df_filtered, station_pick)
    if len(ai_df) == 0:
        st.warning("该范围内暂无数据。")
        return

    valve_opts = ["全部阀门"] + sorted(ai_df["valve_type"].unique())
    valve_pick = st.selectbox("阀门", valve_opts, key="ai_valve")
    if valve_pick != "全部阀门":
        ai_df = ai_df[ai_df["valve_type"] == valve_pick].copy()

    m1, m2, m3 = st.columns(3)
    m1.metric("AI观察异常数", int(ai_df["ai_observe_flag"].sum()))
    m2.metric("AI升级预警数", int(ai_df["ai_escalate_flag"].sum()))
    m3.metric("平均AI分位", f"{ai_df['ai_score_pct'].mean():.1f}" if ai_df["ai_score_pct"].notna().any() else "—")

    trend = (
        ai_df.groupby("date")
        .agg(observe_cnt=("ai_observe_flag", "sum"), escalate_cnt=("ai_escalate_flag", "sum"))
        .reset_index()
        .sort_values("date")
    )
    trend["date_dt"] = pd.to_datetime(trend["date"])

    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    ax.plot(trend["date_dt"], trend["observe_cnt"], marker="o", label="AI观察异常")
    ax.plot(trend["date_dt"], trend["escalate_cnt"], marker="o", label="AI升级预警")
    ax.set_title("近30天AI异常趋势")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    ai_list = ai_df[ai_df["ai_observe_flag"] == True].copy()
    if len(ai_list) == 0:
        st.info("当前范围暂无AI观察异常。若每组样本<30，模型仅做规则风险展示，不触发AI升级。")
    else:
        show_cols = [
            "date",
            "station",
            "valve_type",
            "ai_raw_score",
            "ai_score_pct",
            "ai_observe_flag",
            "ai_escalate_flag",
            "ai_reason_top1",
            "ai_reason_top2",
            "ai_reason_top3",
            "Risk",
            "Risk_final",
        ]
        st.dataframe(ai_list.sort_values("date", ascending=False)[show_cols], use_container_width=True)

def render_tab_dashboard(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame, role: str):
    st.subheader("📊 驾驶舱")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="dash_station")
    else:
        station_pick = STATION_SCOPE

    dash_df = _slice_by_station(df_filtered, station_pick)
    dash_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(dash_df) == 0:
        st.warning("该范围内暂无数据。")
        return

    today = dash_df["date"].max()
    week_start = today - timedelta(days=6)
    prev_week_start = today - timedelta(days=13)
    prev_week_end = today - timedelta(days=7)

    today_high_risk = int(((dash_df["date"] == today) & (dash_df["Risk_final"] == "🔴 高风险")).sum())
    alerts_week = dash_alerts[pd.to_datetime(dash_alerts["date"], errors="coerce").dt.date >= week_start]
    new_alerts_week = len(alerts_week)
    closed_week = int((alerts_week["status"] == "已关闭").sum()) if len(alerts_week) else 0
    close_rate = (closed_week / new_alerts_week * 100) if new_alerts_week else 0

    cur_week_hi = dash_df[dash_df["date"] >= week_start]["HI_final"].mean()
    prev_week_hi = dash_df[(dash_df["date"] >= prev_week_start) & (dash_df["date"] <= prev_week_end)]["HI_final"].mean()
    hi_delta = 0 if np.isnan(cur_week_hi) or np.isnan(prev_week_hi) else cur_week_hi - prev_week_hi

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("当日高风险阀门数", today_high_risk)
    k2.metric("本周新增告警数", new_alerts_week)
    k3.metric("本周闭环率", f"{close_rate:.1f}%")
    k4.metric("平均HI较上周", f"{hi_delta:+.1f}")

    left, right = st.columns([3, 2], gap="small")
    with left:
        if IS_LEADER and station_pick == "全部站点":
            comp = (
                df_filtered.groupby("station")
                .agg(
                    avg_HI=("HI_final", "mean"),
                    red_days=("Risk_final", lambda s: (s == "🔴 高风险").sum()),
                    yellow_days=("Risk_final", lambda s: (s == "🟡 预警").sum()),
                    activity=("Activity", "sum"),
                )
                .reindex(STATIONS)
                .fillna(0)
                .reset_index()
            )
            a2 = alerts_filtered.copy()
            if len(a2) > 0:
                a2["created_dt"] = pd.to_datetime(a2["created_at"], errors="coerce")
                a2["closed_dt"] = pd.to_datetime(a2["closed_at"], errors="coerce")
                a2["close_hours"] = (a2["closed_dt"] - a2["created_dt"]).dt.total_seconds() / 3600
                close_eff = a2.groupby("station")["close_hours"].mean().reindex(STATIONS)
                comp["平均闭环时效(h)"] = comp["station"].map(close_eff).fillna(0).round(1)
            else:
                comp["平均闭环时效(h)"] = 0
            st.dataframe(comp, use_container_width=True)
        else:
            latest = dash_df.sort_values(["valve_type", "date"]).groupby("valve_type").tail(1)
            st.dataframe(
                latest[["date", "station", "valve_type", "HI_final", "Risk_final", "ai_score_pct", "ai_escalate_flag"]],
                use_container_width=True,
            )

    with right:
        daily = (
            dash_df.groupby("date")
            .agg(avg_hi=("HI_final", "mean"), red_cnt=("Risk_final", lambda s: (s == "🔴 高风险").sum()))
            .reset_index()
            .sort_values("date")
        )
        daily["date_dt"] = pd.to_datetime(daily["date"])
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        ax.plot(daily["date_dt"], daily["avg_hi"], marker="o", label="平均HI")
        ax.set_ylim(0, 100)
        ax2 = ax.twinx()
        ax2.bar(daily["date_dt"], daily["red_cnt"], alpha=0.2, color="#d32f2f", label="高风险数")
        ax.set_title("HI与高风险趋势")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=30)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


def render_tab_alerts(alerts_filtered: pd.DataFrame):
    st.subheader("🚨 告警闭环")

    if IS_LEADER:
        station_pick = st.selectbox(
            "站点",
            ["全部站点"] + sorted(alerts_filtered["station"].unique()) if len(alerts_filtered) > 0 else ["全部站点"],
            key="alert_station",
        )
    else:
        station_pick = STATION_SCOPE

    view_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(view_alerts) == 0:
        st.info("当前日期范围无告警。")
        return

    show_cols = [
        "id",
        "date",
        "station",
        "valve_type",
        "risk_level",
        "trigger_source",
        "status",
        "owner",
        "action_taken",
        "verification_result",
        "updated_at",
        "closed_at",
    ]
    st.dataframe(view_alerts[show_cols], use_container_width=True)

    if IS_LEADER:
        st.info("领导账号为只读，不可修改告警状态。")
        return

    work_alerts = view_alerts.copy()
    selected = st.selectbox("选择告警ID", work_alerts["id"].astype(str).tolist(), key="alert_id")
    row = work_alerts[work_alerts["id"].astype(str) == str(selected)].iloc[0]

    cur_status = row["status"] if row["status"] in STATUS_FLOW else "待确认"
    cur_i = STATUS_FLOW.index(cur_status)
    next_options = STATUS_FLOW[cur_i : min(cur_i + 2, len(STATUS_FLOW))]

    cc1, cc2 = st.columns(2)
    with cc1:
        st.text_input("当前状态", value=cur_status, disabled=True, key="alert_cur")
    with cc2:
        new_status = st.selectbox("目标状态", next_options, index=0, key="alert_new")

    action_taken = st.text_area("整改措施（关闭前必填）", value=str(row.get("action_taken", "")), key="alert_action")
    verification_result = st.text_area("复验结果（关闭前必填）", value=str(row.get("verification_result", "")), key="alert_verify")

    if st.button("更新告警状态", use_container_width=True):
        try:
            update_alert_status(
                alert_id=str(selected),
                new_status=new_status,
                operator=st.session_state.user_name,
                action_taken=action_taken,
                verification_result=verification_result,
            )
            st.success("告警状态已更新")
            st.rerun()
        except Exception as ex:
            st.error(f"更新失败：{ex}")


def render_tab_reports(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame):
    st.subheader("📥 报表导出")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="report_station")
    else:
        station_pick = STATION_SCOPE

    rep_df = _slice_by_station(df_filtered, station_pick)
    rep_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(rep_df) == 0:
        st.warning("该范围内暂无可导出数据。")
        return

    exp1, exp2, exp3 = st.columns([1, 1, 1], gap="small")

    with exp1:
        csv_data = rep_df.sort_values(["station", "valve_type", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下载监测数据CSV",
            data=csv_data,
            file_name="psv_data_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp2:
        csv_alert = rep_alerts.sort_values(["station", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下载告警数据CSV",
            data=csv_alert,
            file_name="psv_alerts_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp3:
        ai_rows = rep_df[rep_df["ai_observe_flag"] == True].copy()
        ai_csv = ai_rows[
            [
                "date",
                "station",
                "valve_type",
                "ai_raw_score",
                "ai_score_pct",
                "ai_observe_flag",
                "ai_escalate_flag",
                "ai_reason_top1",
                "ai_reason_top2",
                "ai_reason_top3",
            ]
        ].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下载AI异常明细CSV",
            data=ai_csv,
            file_name="psv_ai_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    avg_hi = rep_df["HI_final"].mean()
    red_cnt = int((rep_df["Risk_final"] == "🔴 高风险").sum())
    yellow_cnt = int((rep_df["Risk_final"] == "🟡 预警").sum())
    ai_obs = int(rep_df["ai_observe_flag"].sum())
    ai_esc = int(rep_df["ai_escalate_flag"].sum())
    close_rate = float((rep_alerts["status"] == "已关闭").mean() * 100) if len(rep_alerts) else 0.0

    top_reasons = (
        rep_df.loc[rep_df["ai_observe_flag"] == True, "ai_reason_top1"]
        .value_counts()
        .head(3)
        .to_dict()
    )
    top_reason_text = "、".join([f"{k}:{v}次" for k, v in top_reasons.items()]) if top_reasons else "暂无"

    summary_lines = [
        "项目名称：基于Isolation Forest算法的LNG储罐安全阀健康监测与风险AI预警系统",
        f"报告范围：{start_date} 至 {end_date}",
        f"账号范围：{station_pick}",
        f"平均HI：{avg_hi:.1f}",
        f"高风险记录数：{red_cnt}",
        f"预警记录数：{yellow_cnt}",
        f"AI观察异常数：{ai_obs}",
        f"AI升级预警数：{ai_esc}",
        f"AI异常主因Top3：{top_reason_text}",
        f"告警闭环率：{close_rate:.1f}%",
        build_leader_storyline(rep_df, rep_alerts),
        "建议：优先处理AI升级预警且连续2天异常的阀门，复验后关闭工单。",
    ]

    summary_text = "\n".join(summary_lines)
    st.text_area("一键周报摘要（可直接贴PPT）", value=summary_text, height=220)
    st.download_button(
        "下载管理摘要TXT",
        data=summary_text.encode("utf-8"),
        file_name="management_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("**最近20条记录**")
    show_cols = [
        "date",
        "station",
        "valve_type",
        "p_now",
        "p_max",
        "level",
        "temp",
        "HI_final",
        "Risk_final",
        "ai_score_pct",
        "ai_observe_flag",
        "ai_escalate_flag",
    ]
    st.dataframe(rep_df.sort_values("date", ascending=False)[show_cols].head(20), use_container_width=True)


# ================== Top Tabs ==================
# 默认首开第一个Tab：历史分析
hist_tab, ai_tab, dash_tab, alert_tab, report_tab = st.tabs(
    ["历史分析", "AI预警中心", "驾驶舱", "告警闭环", "报表导出"]
)

with hist_tab:
    render_tab_history(df_f, alerts_f)

with ai_tab:
    render_tab_ai(df_f)

with dash_tab:
    render_tab_dashboard(df_f, alerts_f, ROLE)

with alert_tab:
    render_tab_alerts(alerts_f)

with report_tab:
    render_tab_reports(df_f, alerts_f)
