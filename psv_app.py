import os
import uuid
from datetime import timedelta

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


SUPABASE_URL = _secret_get("SUPABASE_URL", "https://ynowvxcsvjskwkeauvkz.supabase.co") or os.getenv("SUPABASE_URL", "https://ynowvxcsvjskwkeauvkz.supabase.co")
SUPABASE_KEY = _secret_get("SUPABASE_KEY", "sb_publishable_aezshZPqB78WBtyWtTf8Tg_UVpCEZzd") or os.getenv("SUPABASE_KEY", "sb_publishable_aezshZPqB78WBtyWtTf8Tg_UVpCEZzd")

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
st.set_page_config(page_title="LNG安全阀多站点监测系统", layout="wide")


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
        alerts.loc[idx, "risk_level"] = record.get("risk_level", alerts.loc[idx, "risk_level"])
        alerts.loc[idx, "trigger_source"] = record.get("trigger_source", alerts.loc[idx, "trigger_source"])
        alerts.loc[idx, "trigger_detail"] = str(record.get("trigger_detail", alerts.loc[idx, "trigger_detail"]))
        alerts.loc[idx, "updated_at"] = now_iso
        alerts.loc[idx, "status"] = keep_status
    else:
        new_alert = {
            "id": str(uuid.uuid4()),
            "date": str(record["date"]),
            "station": record["station"],
            "valve_type": record["valve_type"],
            "risk_level": record.get("risk_level", "🔴 高风险"),
            "trigger_source": record.get("trigger_source", "rule"),
            "trigger_detail": str(record.get("trigger_detail", "")),
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
        rows = alerts.to_dict(orient="records")
        supabase.table(TABLE_ALERT).upsert(rows, on_conflict="id").execute()
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
        supabase.table(TABLE_ALERT).upsert(alerts.to_dict(orient="records"), on_conflict="id").execute()
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
def compute_scores(df0: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return df0

    df = _normalize_df(df0).copy()
    df = df.sort_values(["station", "valve_type", "date"]).reset_index(drop=True)

    df["ratio"] = df["p_max"] / SET_P
    df["slope"] = (
        df.groupby(["station", "valve_type"])["p_max"]
        .apply(lambda s: s.diff().rolling(3).mean())
        .reset_index(level=[0, 1], drop=True)
    )

    hi = np.full(len(df), 100.0)
    hi -= np.where(df["ratio"] >= 1.00, 35, 0)
    hi -= np.where((df["ratio"] >= 0.98) & (df["ratio"] < 1.00), 20, 0)
    hi -= np.where((df["ratio"] >= 0.95) & (df["ratio"] < 0.98), 10, 0)

    hi -= np.where(df["slope"] > 0.01, 10, 0)
    hi -= np.where(df["slope"] > 0.02, 10, 0)

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

    def risk(x: float) -> str:
        if x >= 85:
            return "🟢 安全"
        if x >= 70:
            return "🟡 预警"
        return "🔴 高风险"

    df["Risk"] = df["HI"].apply(risk)
    df["Activity"] = df.get("psv_act", 0).fillna(0) + df.get("psv_weeping", 0).fillna(0)

    df["AI_anomaly"] = False
    df["AI_score"] = np.nan

    if enable_ai and SKLEARN_OK:
        features = ["p_now", "p_max", "level", "temp", "ratio", "slope", "Activity"]
        for _, g in df.groupby(["station", "valve_type"]):
            idx = g.index
            if len(g) < 10:
                continue
            x = g[features].copy().apply(pd.to_numeric, errors="coerce")
            x = x.fillna(x.median(numeric_only=True))

            scaler = StandardScaler()
            xs = scaler.fit_transform(x.values)

            iso = IsolationForest(n_estimators=200, contamination=float(contamination), random_state=42)
            iso.fit(xs)
            pred = iso.predict(xs)
            score = -iso.score_samples(xs)

            df.loc[idx, "AI_anomaly"] = pred == -1
            df.loc[idx, "AI_score"] = score

        df["HI_final"] = np.clip(df["HI"] - df["AI_anomaly"].astype(int) * 10, 0, 100)
    else:
        df["HI_final"] = df["HI"]

    df["Risk_final"] = df["HI_final"].apply(risk)
    return df


def _calc_trigger_source(row: pd.Series):
    rule_hit = str(row.get("Risk_final", "")) == "🔴 高风险"
    ai_hit = bool(row.get("AI_anomaly", False))

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
                    "AI_anomaly": bool(row.get("AI_anomaly", False)),
                },
            }
        )


# ================== UI ==================
st.title("LNG安全阀多站点AI健康监测与告警闭环系统")
st.caption("一期：华盘站/罗所站/领导三账号，按站点数据隔离，领导只读。")

st.sidebar.divider()
st.sidebar.header("🧠 AI 异常检测")
enable_ai = st.sidebar.checkbox("启用 AI 异常检测", value=True)
contamination = st.sidebar.slider("异常比例（越大越敏感）", min_value=0.02, max_value=0.20, value=0.08, step=0.01)
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

# Common date filter
min_d, max_d = df["date"].min(), df["date"].max()
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    start_date = st.date_input("开始日期", value=min_d, min_value=min_d, max_value=max_d, key="start")
with c2:
    end_date = st.date_input("结束日期", value=max_d, min_value=min_d, max_value=max_d, key="end")
with c3:
    st.caption("建议：汇报场景优先选择最近30天。")

df_f = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
alerts_f = alerts[
    (pd.to_datetime(alerts["date"], errors="coerce").dt.date >= start_date)
    & (pd.to_datetime(alerts["date"], errors="coerce").dt.date <= end_date)
].copy()

if IS_LEADER:
    tab_dashboard, tab_alert, tab_export, tab_history = st.tabs(["领导驾驶舱", "告警中心", "报表导出", "历史分析"])
else:
    tab_dashboard, tab_alert, tab_export, tab_history = st.tabs(["站点工作台", "告警中心", "报表导出", "历史分析"])

with tab_dashboard:
    if IS_LEADER:
        st.subheader("📊 领导驾驶舱")
        today = df_f["date"].max()
        week_start = today - timedelta(days=6)
        prev_week_start = today - timedelta(days=13)
        prev_week_end = today - timedelta(days=7)

        today_high_risk = int(((df_f["date"] == today) & (df_f["Risk_final"] == "🔴 高风险")).sum())
        alerts_week = alerts_f[pd.to_datetime(alerts_f["date"]).dt.date >= week_start]
        new_alerts_week = len(alerts_week)
        closed_week = int((alerts_week["status"] == "已关闭").sum()) if len(alerts_week) else 0
        close_rate = (closed_week / new_alerts_week * 100) if new_alerts_week else 0

        cur_week_hi = df_f[(df_f["date"] >= week_start)]["HI_final"].mean()
        prev_week_hi = df_f[(df_f["date"] >= prev_week_start) & (df_f["date"] <= prev_week_end)]["HI_final"].mean()
        hi_delta = 0 if np.isnan(cur_week_hi) or np.isnan(prev_week_hi) else cur_week_hi - prev_week_hi

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("当日高风险阀门数", today_high_risk)
        k2.metric("本周新增告警数", new_alerts_week)
        k3.metric("本周闭环率", f"{close_rate:.1f}%")
        k4.metric("平均HI较上周", f"{hi_delta:+.1f}")

        st.markdown("**华盘 vs 罗所 对比**")
        comp = (
            df_f.groupby("station")
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

        a2 = alerts_f.copy()
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
        st.subheader("📌 站点工作台（最新状态）")
        latest = df_f.sort_values(["valve_type", "date"]).groupby("valve_type").tail(1)
        cols = st.columns(3)
        for i, valve in enumerate(["泵后安全阀", "储罐主阀", "储罐辅阀"]):
            block = latest[latest["valve_type"] == valve]
            if len(block) == 0:
                cols[i].metric(f"{valve} HI", "—")
                cols[i].metric("风险", "—")
                continue
            row = block.iloc[0]
            cols[i].metric(f"{valve} HI", f"{row['HI_final']:.1f}")
            cols[i].metric("风险", row["Risk_final"])
            cols[i].caption(f"压力占整定：{row['ratio'] * 100:.1f}%")

with tab_alert:
    st.subheader("🚨 告警中心")
    if len(alerts_f) == 0:
        st.info("当前日期范围无告警。")
    else:
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
            "created_at",
            "updated_at",
            "closed_at",
        ]
        st.dataframe(alerts_f[show_cols], use_container_width=True)

    if not IS_LEADER and len(alerts_f) > 0:
        st.markdown("**处理告警**")
        work_alerts = alerts_f.copy()
        selected = st.selectbox(
            "选择告警ID",
            work_alerts["id"].astype(str).tolist(),
            index=0,
        )
        row = work_alerts[work_alerts["id"].astype(str) == str(selected)].iloc[0]

        cur_status = row["status"] if row["status"] in STATUS_FLOW else "待确认"
        cur_i = STATUS_FLOW.index(cur_status)
        next_options = STATUS_FLOW[cur_i : min(cur_i + 2, len(STATUS_FLOW))]

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("当前状态", value=cur_status, disabled=True)
        with c2:
            new_status = st.selectbox("目标状态", next_options, index=0)

        action_taken = st.text_area("整改措施（关闭前必填）", value=str(row.get("action_taken", "")))
        verification_result = st.text_area("复验结果（关闭前必填）", value=str(row.get("verification_result", "")))

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
    elif IS_LEADER:
        st.info("领导账号为只读，不可修改告警状态。")

with tab_export:
    st.subheader("📥 报表导出")

    csv_data = df_f.sort_values(["station", "valve_type", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下载监测数据CSV",
        data=csv_data,
        file_name="psv_data_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    csv_alert = alerts_f.sort_values(["station", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "下载告警数据CSV",
        data=csv_alert,
        file_name="psv_alerts_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

    avg_hi = df_f["HI_final"].mean()
    red_cnt = int((df_f["Risk_final"] == "🔴 高风险").sum())
    yellow_cnt = int((df_f["Risk_final"] == "🟡 预警").sum())
    close_rate = 0.0
    if len(alerts_f) > 0:
        close_rate = (alerts_f["status"] == "已关闭").mean() * 100

    summary_lines = [
        f"报告范围：{start_date} 至 {end_date}",
        f"账号范围：{STATION_SCOPE}",
        f"平均HI：{avg_hi:.1f}",
        f"高风险记录数：{red_cnt}",
        f"预警记录数：{yellow_cnt}",
        f"告警闭环率：{close_rate:.1f}%",
    ]

    if IS_LEADER:
        comp = (
            df_f.groupby("station")["HI_final"]
            .mean()
            .reindex(STATIONS)
            .fillna(0)
        )
        summary_lines.append(f"站点对比：华盘平均HI={comp.get(STATIONS[0], 0):.1f}，罗所平均HI={comp.get(STATIONS[1], 0):.1f}")

    summary_text = "\n".join(summary_lines)
    st.text_area("管理摘要（可直接贴PPT）", value=summary_text, height=180)
    st.download_button(
        "下载管理摘要TXT",
        data=summary_text.encode("utf-8"),
        file_name="management_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

with tab_history:
    st.subheader("📈 历史分析")

    station_opts = sorted(df_f["station"].unique())
    if IS_LEADER:
        station_pick = st.selectbox("站点", station_opts, index=0)
    else:
        station_pick = STATION_SCOPE
        st.info(f"当前站点：{station_pick}")

    sdf = df_f[df_f["station"] == station_pick].copy()
    if len(sdf) == 0:
        st.warning("该站点当前筛选范围内无数据。")
        st.stop()

    valve_pick = st.selectbox("阀门", sorted(sdf["valve_type"].unique()), index=0)
    vdf = sdf[sdf["valve_type"] == valve_pick].sort_values("date").copy()
    vdf["date_dt"] = pd.to_datetime(vdf["date"])

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(vdf["date_dt"], vdf["p_max"], marker="o", label="p_max")
        ax.plot(vdf["date_dt"], vdf["p_now"], marker="o", linestyle="--", label="p_now")
        ax.axhline(SET_P, linestyle="--", label="整定压力 1.32MPa")
        ax.set_title(f"{station_pick} - {valve_pick} 压力趋势")
        ax.set_ylabel("MPa")
        ax.legend()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        ax.plot(vdf["date_dt"], vdf["HI_final"], marker="o")
        ax.set_title(f"{station_pick} - {valve_pick} 健康指数趋势")
        ax.set_ylabel("HI")
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.xticks(rotation=30)
        st.pyplot(fig)

    st.markdown("**最近20条记录**")
    show_cols = [
        "date",
        "station",
        "valve_type",
        "p_now",
        "p_max",
        "level",
        "temp",
        "psv_act",
        "psv_weeping",
        "HI_final",
        "Risk_final",
        "AI_anomaly",
        "AI_score",
    ]
    st.dataframe(sdf.sort_values("date", ascending=False)[show_cols].head(20), use_container_width=True)
