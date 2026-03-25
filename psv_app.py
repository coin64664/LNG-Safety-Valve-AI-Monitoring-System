
import os
import uuid
from datetime import timedelta
from typing import Tuple
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg

import matplotlib as mpl
from matplotlib import font_manager

from ai_engine import (
    SKLEARN_OK,
    build_case_replay,
    build_model_vote_matrix,
    build_validation_summary,
    run_scoring_pipeline,
)
from data_pipeline import (
    BASE_ALERT_COLS,
    BASE_AUDIT_COLS,
    BASE_DATA_COLS,
    DEFAULT_STATION,
    STATUS_FLOW,
    STATIONS,
    append_audit as dp_append_audit,
    create_or_update_alert as dp_create_or_update_alert,
    ensure_local_files as dp_ensure_local_files,
    list_alerts as dp_list_alerts,
    load_data as dp_load_data,
    normalize_data_df as dp_normalize_df,
    save_record as dp_save_record,
    scope_filter as dp_scope_filter,
    update_alert_status as dp_update_alert_status,
)
from reporting import (
    build_data_quality_table,
    generate_management_summary,
)
from risk_engine import SET_P, risk_from_hi as engine_risk_from_hi


# ================== Config ==================
DATA_FILE = "psv_data.csv"
ALERT_FILE = "psv_alerts.csv"
AUDIT_FILE = "psv_audit_logs.csv"

TABLE_DATA = "psv_data"
TABLE_ALERT = "psv_alerts"
TABLE_AUDIT = "psv_audit_logs"

DEFAULT_BRAND_PRIMARY = "#0B5ED7"
DEFAULT_BRAND_ACCENT = "#E65100"
LOGO_FILE = "company_logo.png"


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


def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    c = hex_color.strip().lstrip("#")
    if len(c) != 6:
        return 11, 94, 215
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _mix_hex(base_hex: str, target_hex: str, ratio: float) -> str:
    br, bg, bb = _hex_to_rgb(base_hex)
    tr, tg, tb = _hex_to_rgb(target_hex)
    r = br * (1 - ratio) + tr * ratio
    g = bg * (1 - ratio) + tg * ratio
    b = bb * (1 - ratio) + tb * ratio
    return _rgb_to_hex((r, g, b))


def load_logo_and_palette(logo_path: str = LOGO_FILE) -> dict:
    palette = {
        "primary": DEFAULT_BRAND_PRIMARY,
        "secondary": _mix_hex(DEFAULT_BRAND_PRIMARY, "#FFFFFF", 0.78),
        "primary_dark": _mix_hex(DEFAULT_BRAND_PRIMARY, "#000000", 0.22),
        "accent": DEFAULT_BRAND_ACCENT,
        "bg_top": "#EEF5FF",
        "bg_bottom": "#F8FBFF",
        "logo_path": logo_path,
        "logo_exists": Path(logo_path).exists(),
    }
    if not palette["logo_exists"]:
        return palette

    try:
        img = mpimg.imread(logo_path)
        if img.ndim < 3:
            return palette
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        if arr.shape[2] == 4:
            alpha = arr[:, :, 3] > 25
            arr = arr[:, :, :3]
            arr = arr[alpha]
        else:
            arr = arr[:, :, :3].reshape(-1, 3)

        if arr.ndim != 2 or len(arr) == 0:
            return palette

        if len(arr) > 12000:
            pick = np.linspace(0, len(arr) - 1, 12000).astype(int)
            arr = arr[pick]

        quant = (arr // 16) * 16
        uniq, cnt = np.unique(quant, axis=0, return_counts=True)
        dom = uniq[cnt.argmax()]
        primary = _rgb_to_hex((int(dom[0]), int(dom[1]), int(dom[2])))
        palette["primary"] = primary
        palette["secondary"] = _mix_hex(primary, "#FFFFFF", 0.80)
        palette["primary_dark"] = _mix_hex(primary, "#000000", 0.18)
        palette["bg_top"] = _mix_hex(primary, "#FFFFFF", 0.90)
        palette["bg_bottom"] = _mix_hex(primary, "#FFFFFF", 0.97)
    except Exception:
        return palette

    return palette


def _load_logo_for_display(logo_path: str):
    if not Path(logo_path).exists():
        return None
    try:
        img = mpimg.imread(logo_path)
        if img.ndim < 3:
            return img
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

        rgb = arr[:, :, :3]
        if arr.shape[2] == 4:
            mask = arr[:, :, 3] > 12
        else:
            mask = np.any(rgb < 245, axis=2)

        if not mask.any():
            return img

        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        pad = max(2, int(0.03 * max(y1 - y0, x1 - x0)))
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(arr.shape[0], y1 + pad)
        x1 = min(arr.shape[1], x1 + pad)
        cropped = arr[y0:y1, x0:x1]
        if cropped.dtype == np.uint8:
            return cropped.astype(np.float32) / 255.0
        return cropped
    except Exception:
        return None


def inject_ui_theme(palette: dict, enabled: bool = True):
    if not enabled:
        return

    css = f"""
    <style>
    :root {{
      --brand-primary: {palette['primary']};
      --brand-secondary: {palette['secondary']};
      --brand-dark: {palette['primary_dark']};
      --brand-accent: {palette['accent']};
      --bg-top: {palette['bg_top']};
      --bg-bottom: {palette['bg_bottom']};
    }}

    .stApp {{
      background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 48%, #ffffff 100%);
    }}

    header[data-testid="stHeader"] {{
      display: none !important;
      height: 0 !important;
    }}

    [data-testid="stToolbar"] {{
      display: none !important;
    }}

    [data-testid="stDecoration"] {{
      display: none !important;
    }}

    .block-container {{
      padding-top: .6rem;
      padding-bottom: 1.8rem;
    }}

    section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #F2F7FF 0%, #FFFFFF 100%);
      border-right: 1px solid rgba(11, 94, 215, 0.10);
    }}

    div[data-testid="stMetric"] {{
      border: 1px solid rgba(11, 94, 215, 0.14);
      border-radius: 12px;
      padding: 10px 14px;
      background: #FFFFFF;
      box-shadow: 0 8px 24px rgba(17, 65, 133, 0.08);
    }}

    div[data-baseweb="tab-list"] {{
      gap: 10px;
    }}

    div[data-baseweb="tab-list"] button {{
      border: 1px solid rgba(11, 94, 215, 0.18) !important;
      border-radius: 999px !important;
      background: #ffffff !important;
      color: #1c2b42 !important;
      padding: 8px 16px !important;
      font-weight: 600 !important;
    }}

    div[data-baseweb="tab-list"] button[aria-selected="true"] {{
      background: linear-gradient(90deg, var(--brand-primary), var(--brand-dark)) !important;
      color: #ffffff !important;
      border-color: transparent !important;
      box-shadow: 0 6px 16px rgba(11, 94, 215, 0.28);
    }}

    button[kind="primary"], .stButton > button {{
      border-radius: 10px !important;
      border: 1px solid rgba(11, 94, 215, 0.25) !important;
    }}

    .stButton > button[data-testid="baseButton-primary"] {{
      background: linear-gradient(120deg, var(--brand-primary), var(--brand-dark)) !important;
      color: #ffffff !important;
    }}

    div[data-testid="stDataFrame"] {{
      border: 1px solid rgba(11, 94, 215, 0.16);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 6px 18px rgba(17, 65, 133, 0.06);
    }}

    div[data-testid="stAlert"] {{
      border-radius: 12px;
      border: 1px solid rgba(11, 94, 215, 0.14);
      box-shadow: 0 4px 14px rgba(17, 65, 133, 0.06);
    }}

    .brand-header {{
      border: 1px solid rgba(11, 94, 215, 0.16);
      border-radius: 16px;
      background: linear-gradient(120deg, #FFFFFF 0%, var(--brand-secondary) 100%);
      padding: 10px 14px;
      margin-bottom: 10px;
      box-shadow: 0 10px 26px rgba(17, 65, 133, 0.10);
    }}

    .brand-title {{
      margin: 0;
      font-size: clamp(1.35rem, 2.2vw, 2.25rem);
      line-height: 1.22;
      font-weight: 800;
      letter-spacing: .2px;
      word-break: break-word;
      background: linear-gradient(90deg, var(--brand-dark), var(--brand-primary));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    .brand-subtitle {{
      margin-top: .30rem;
      color: #3A567A;
      font-size: .98rem;
      font-weight: 500;
    }}

    .brand-chip {{
      display: inline-block;
      margin-top: .45rem;
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(11, 94, 215, 0.11);
      color: #17427A;
      font-size: .82rem;
      font-weight: 700;
    }}

    .hero-panel, .evidence-panel {{
      border: 1px solid rgba(11, 94, 215, 0.14);
      border-radius: 18px;
      background: linear-gradient(145deg, rgba(255,255,255,.96) 0%, rgba(243,248,255,.92) 100%);
      box-shadow: 0 12px 28px rgba(17, 65, 133, 0.08);
      padding: 16px 18px;
      margin-bottom: 12px;
    }}

    .hero-kicker {{
      font-size: .82rem;
      font-weight: 700;
      letter-spacing: .8px;
      text-transform: uppercase;
      color: #486a92;
      margin-bottom: 6px;
    }}

    .hero-headline {{
      font-size: clamp(1.2rem, 1.9vw, 1.9rem);
      line-height: 1.3;
      color: #18385c;
      font-weight: 800;
      margin-bottom: 6px;
    }}

    .hero-desc, .evidence-text {{
      color: #3f5d80;
      line-height: 1.65;
      font-size: .95rem;
    }}

    .signal-card {{
      border: 1px solid rgba(11, 94, 215, 0.14);
      border-radius: 16px;
      background: linear-gradient(160deg, #ffffff 0%, #f6faff 100%);
      box-shadow: 0 8px 22px rgba(17, 65, 133, 0.06);
      padding: 14px 16px;
      min-height: 128px;
    }}

    .signal-label {{
      color: #5a7ca4;
      font-size: .84rem;
      font-weight: 700;
      margin-bottom: 4px;
    }}

    .signal-value {{
      color: #18385c;
      font-size: 1.6rem;
      font-weight: 800;
      line-height: 1.15;
      margin-bottom: 6px;
    }}

    .signal-note {{
      color: #446483;
      font-size: .92rem;
      line-height: 1.55;
    }}

    .section-note {{
      padding: 10px 12px;
      border-radius: 12px;
      border-left: 4px solid rgba(11, 94, 215, 0.78);
      background: rgba(11, 94, 215, 0.08);
      color: #21476f;
      margin: 8px 0 14px 0;
      font-size: .92rem;
      line-height: 1.6;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_brand_header(logo_path: str, palette: dict, enabled: bool = True):
    if not enabled:
        st.title("LNG储罐安全阀智能预警系统")
        st.caption("规则机理｜自适应基线｜时序退化｜异常共识")
        return

    left, right = st.columns([1, 9], gap="small")
    with left:
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        if Path(logo_path).exists():
            logo_img = _load_logo_for_display(logo_path)
            if logo_img is not None:
                st.image(logo_img, width=96)
            else:
                st.image(logo_path, width=96)
        else:
            st.markdown(
                "<div style='height:82px;display:flex;align-items:center;justify-content:center;"
                "border:1px dashed rgba(11,94,215,.35);border-radius:12px;color:#1b4f8c;font-size:.85rem;'>公司Logo</div>",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            "<div class='brand-header'>"
            "<h1 class='brand-title'>LNG储罐安全阀健康监测与风险预警系统</h1>"
            "<div class='brand-subtitle'>多站点分角色管理 + 历史趋势分析 + 多模型异常识别 + 告警闭环跟踪</div>"
            "<div class='brand-chip'>运行监测平台</div>"
            "</div>",
            unsafe_allow_html=True,
        )


def apply_plot_theme(palette: dict, enabled: bool = True):
    if not enabled:
        return
    mpl.rcParams["axes.facecolor"] = "#FFFFFF"
    mpl.rcParams["figure.facecolor"] = "#FFFFFF"
    mpl.rcParams["axes.edgecolor"] = "#7f93b3"
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.20
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["grid.color"] = _mix_hex(palette["primary"], "#000000", 0.30)
    mpl.rcParams["axes.titleweight"] = "bold"
    mpl.rcParams["axes.titlecolor"] = "#1a3457"


SUPABASE_URL = _secret_get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = _secret_get("SUPABASE_KEY", "") or os.getenv("SUPABASE_KEY", "")

USE_SUPABASE = True
supabase = None
if USE_SUPABASE and SUPABASE_OK and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
elif USE_SUPABASE:
    st.sidebar.warning("⚠️ 未检测到 Supabase 配置（SUPABASE_URL / SUPABASE_KEY），将回退为本地CSV存储。")
    USE_SUPABASE = False


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
st.set_page_config(page_title="LNG储罐安全阀风险预警系统", layout="wide")

ENABLE_UI_THEME = str(_secret_get("ENABLE_UI_THEME", "1") or os.getenv("ENABLE_UI_THEME", "1")).strip().lower() not in {"0", "false", "no", "off"}
COMPANY_LOGO_PATH = str(_secret_get("COMPANY_LOGO_PATH", LOGO_FILE) or os.getenv("COMPANY_LOGO_PATH", LOGO_FILE))
BRAND = load_logo_and_palette(COMPANY_LOGO_PATH)
inject_ui_theme(BRAND, enabled=ENABLE_UI_THEME)
apply_plot_theme(BRAND, enabled=ENABLE_UI_THEME)


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
    render_brand_header(COMPANY_LOGO_PATH, BRAND, enabled=ENABLE_UI_THEME)
    st.info("请输入账号和密码后进入系统。")
    st.markdown(
        """
        <div style="
            margin-top: 10px;
            max-width: 820px;
            border: 1px solid rgba(11,94,215,.18);
            border-radius: 14px;
            background: linear-gradient(120deg, #ffffff 0%, #f3f8ff 100%);
            padding: 14px 16px;
            box-shadow: 0 8px 20px rgba(17,65,133,.08);
            color: #1f3b64;
        ">
            <div style="font-size: 1.04rem; font-weight: 800; margin-bottom: 6px;">版本信息：v0.5 产品成品版</div>
            <div style="font-size: .95rem; line-height: 1.62;">
                更新内容：<br/>
                1. 首页恢复为历史分析优先，突出压力趋势、热力图和重点阀门状态；<br/>
                2. AI引擎升级为多模型融合，新增局部异常识别、时序突变检测和统一风险共识；<br/>
                3. 系统强化健康档案、处置建议和告警闭环展示，提升现场使用体验。
            </div>
            <div style="margin-top: 8px; font-size: .92rem; font-weight: 700;">开发：杨翔允</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    dp_ensure_local_files(DATA_FILE, ALERT_FILE, AUDIT_FILE)


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
    return dp_normalize_df(df0, stations=STATIONS, default_station=DEFAULT_STATION)


def _scope_filter(df: pd.DataFrame, station_scope: str) -> pd.DataFrame:
    return dp_scope_filter(df, station_scope)


def load_data(station_scope: str, role: str) -> pd.DataFrame:
    return dp_load_data(
        station_scope,
        role,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_data=TABLE_DATA,
        data_file=DATA_FILE,
    )


def _write_local_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def save_record(record: dict, station_scope: str, role: str) -> None:
    dp_save_record(
        record,
        station_scope,
        role,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_data=TABLE_DATA,
        data_file=DATA_FILE,
    )

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
    dp_append_audit(
        entity_type,
        entity_id,
        action,
        operator,
        payload,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_audit=TABLE_AUDIT,
        audit_file=AUDIT_FILE,
    )


def _find_alert(alerts: pd.DataFrame, date_value, station: str, valve_type: str):
    m = (
        (alerts["date"] == pd.to_datetime(date_value).date())
        & (alerts["station"] == station)
        & (alerts["valve_type"] == valve_type)
    )
    return alerts[m]


def create_or_update_alert(record: dict):
    dp_create_or_update_alert(
        record,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_alert=TABLE_ALERT,
        alert_file=ALERT_FILE,
    )


def update_alert_status(alert_id: str, new_status: str, operator: str, action_taken: str = "", verification_result: str = ""):
    dp_update_alert_status(
        alert_id,
        new_status,
        operator,
        action_taken,
        verification_result,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_alert=TABLE_ALERT,
        table_audit=TABLE_AUDIT,
        alert_file=ALERT_FILE,
        audit_file=AUDIT_FILE,
    )


def list_alerts(station_scope: str, role: str) -> pd.DataFrame:
    return dp_list_alerts(
        station_scope,
        role,
        use_supabase=USE_SUPABASE,
        supabase=supabase,
        table_alert=TABLE_ALERT,
        alert_file=ALERT_FILE,
    )


# ================== Scoring ==================
def _risk_from_hi(x: float) -> str:
    return engine_risk_from_hi(x)



def compute_scores(df0: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return df0
    return run_scoring_pipeline(_normalize_df(df0), enable_ai=enable_ai, contamination=contamination)


def _calc_trigger_source(row: pd.Series):
    rule_hit = str(row.get("Risk_final", "")) == "🔴 高风险"
    ai_hit = str(row.get("risk_stage", "")) in ["AI升级", "AI高风险"] or bool(row.get("ai_escalate_flag", False))

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
                    "baseline_dev_score": float(row.get("baseline_dev_score", np.nan)) if pd.notna(row.get("baseline_dev_score", np.nan)) else None,
                    "degradation_score": float(row.get("degradation_score", np.nan)) if pd.notna(row.get("degradation_score", np.nan)) else None,
                    "ai_if_score": float(row.get("ai_if_score", np.nan)) if pd.notna(row.get("ai_if_score", np.nan)) else None,
                    "ai_lof_score": float(row.get("ai_lof_score", np.nan)) if pd.notna(row.get("ai_lof_score", np.nan)) else None,
                    "ai_shift_score": float(row.get("ai_shift_score", np.nan)) if pd.notna(row.get("ai_shift_score", np.nan)) else None,
                    "ai_degradation_score": float(row.get("ai_degradation_score", np.nan)) if pd.notna(row.get("ai_degradation_score", np.nan)) else None,
                    "ai_vote_count": int(row.get("ai_vote_count", 0)) if pd.notna(row.get("ai_vote_count", 0)) else 0,
                    "ai_confidence": float(row.get("ai_confidence", np.nan)) if pd.notna(row.get("ai_confidence", np.nan)) else None,
                    "consensus_score": float(row.get("consensus_score", np.nan)) if pd.notna(row.get("consensus_score", np.nan)) else None,
                    "ai_score_pct": float(row.get("ai_score_pct", np.nan)) if pd.notna(row.get("ai_score_pct", np.nan)) else None,
                    "risk_stage": row.get("risk_stage", "正常"),
                    "risk_reason_path": row.get("risk_reason_path", ""),
                    "action_suggestion": row.get("action_suggestion", ""),
                    "ai_reason_top1": row.get("ai_reason_top1", "-"),
                    "ai_reason_top2": row.get("ai_reason_top2", "-"),
                    "ai_reason_top3": row.get("ai_reason_top3", "-"),
                },
            }
        )

# ================== UI ==================
render_brand_header(COMPANY_LOGO_PATH, BRAND, enabled=ENABLE_UI_THEME)

st.sidebar.divider()
st.sidebar.header("🧠 AI 参数")
enable_ai = st.sidebar.checkbox("启用多模型异常识别", value=True)
contamination = st.sidebar.slider("异常灵敏度（越大越敏感）", min_value=0.02, max_value=0.20, value=0.08, step=0.01)
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
    data_source_tag = st.sidebar.selectbox("数据来源标记", ["真实数据", "模拟数据", "未标注"], index=0)

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
                    "data_source_tag": data_source_tag,
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

validation_summary = build_validation_summary(df_f, alerts_f)
quality_table = build_data_quality_table(df_f)


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
        return "当前时间范围暂无运行数据。"

    comp = build_hi_compare(df_filtered)
    worst_name = "-"
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
        trend_text = "近7天平均HI暂无可比数据"
    else:
        delta = recent7 - prev7
        trend_text = f"近7天平均HI较前7天{'上升' if delta >= 0 else '下降'} {abs(delta):.1f}"

    ai_obs = int(df_filtered["risk_stage"].eq("AI观察").sum()) if "risk_stage" in df_filtered.columns else 0
    ai_esc = int(df_filtered["risk_stage"].isin(["AI升级", "AI高风险"]).sum()) if "risk_stage" in df_filtered.columns else 0

    close_rate = 0.0
    if len(alerts_filtered) > 0:
        close_rate = float((alerts_filtered["status"] == "已关闭").mean() * 100)

    return (
        f"重点关注阀门：{worst_name}，平均HI {worst_hi:.1f}；{trend_text}；"
        f"AI观察 {ai_obs} 次，AI升级/高风险 {ai_esc} 次；告警闭环率 {close_rate:.1f}%。"
    )


def build_valve_health_profile(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame, valve_pick: str) -> dict:
    valve_df = df_filtered[df_filtered["valve_type"] == valve_pick].copy().sort_values("date")
    valve_alerts = alerts_filtered[alerts_filtered["valve_type"] == valve_pick].copy() if len(alerts_filtered) > 0 else pd.DataFrame()
    if len(valve_df) == 0:
        return {}

    latest = valve_df.iloc[-1]
    return {
        "valve_type": valve_pick,
        "latest_date": str(latest["date"]),
        "latest_hi": float(latest.get("HI_final", np.nan)),
        "latest_stage": str(latest.get("risk_stage", "正常")),
        "latest_risk": str(latest.get("Risk_final", "🟢 安全")),
        "observe_days": int(valve_df["risk_stage"].isin(["AI观察", "AI升级", "AI高风险"]).sum()),
        "upgrade_days": int(valve_df["risk_stage"].isin(["AI升级", "AI高风险"]).sum()),
        "alert_count": int(len(valve_alerts)),
        "last_alert_status": str(valve_alerts.sort_values("date").iloc[-1]["status"]) if len(valve_alerts) > 0 else "暂无告警",
        "action_suggestion": str(latest.get("action_suggestion", "当前运行平稳，建议按既定频次巡检。")),
        "risk_reason_path": str(latest.get("risk_reason_path", "")),
    }


def render_tab_history(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame):
    st.subheader("历史分析")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="hist_station")
    else:
        station_pick = STATION_SCOPE
        st.info(f"当前站点：{station_pick}")

    hist_df = _slice_by_station(df_filtered, station_pick)
    hist_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(hist_df) == 0:
        st.warning("当前范围暂无历史数据。")
        return

    valve_opts = sorted(hist_df["valve_type"].unique())
    valve_pick = st.selectbox("阀门", valve_opts, key="hist_valve")
    vdf = build_pressure_trend(hist_df, "全部站点" if station_pick == "全部站点" else station_pick, valve_pick)

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        st.markdown("**3线压力趋势图**")
        if len(vdf) == 0:
            st.info("当前阀门暂无趋势数据。")
        else:
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            ax.plot(vdf["date_dt"], vdf["p_now"], marker="o", linestyle="--", label="p_now")
            ax.plot(vdf["date_dt"], vdf["p_max"], marker="o", label="p_max")
            ax.axhline(SET_P, linestyle="-.", color="#6d4c41", label=f"整定线 {SET_P:.2f}MPa")

            ai_points = vdf[vdf["ai_observe_flag"] == True]
            if len(ai_points) > 0:
                ax.scatter(ai_points["date_dt"], ai_points["p_max"], color="#d32f2f", zorder=4, label="AI观察点")

            ax.set_ylabel("MPa")
            ax.set_title("压力趋势（3线）")
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
                trend_word = "整体上升"
            elif pmax_delta < -0.02:
                trend_word = "有所回落"
            else:
                trend_word = "基本平稳"

            st.caption(
                "趋势解读："
                f"p_max 由 {pmax_start:.2f}MPa 变化到 {pmax_end:.2f}MPa，当前{trend_word}，累计变化 {pmax_delta:+.2f}MPa；"
                f"接近整定压力 {SET_P * 0.95:.2f}MPa 共 {near_set_days} 天，达到或超过整定压力共 {exceed_set_days} 天；"
                f"AI观察点共出现 {ai_days} 天。"
            )
            st.markdown(
                "<div class='section-note'>趋势图用于判断压力是否持续上行、是否频繁接近整定压力，以及AI观察点是否与压力波动同步出现。</div>",
                unsafe_allow_html=True,
            )
            if exceed_set_days > 0:
                st.warning("已出现达到或超过整定压力的记录，建议优先复核阀门状态和工况变化。")
            elif near_set_days > 0:
                st.info("近期存在接近整定压力的天数，建议结合液位、温度和动作记录提高巡检频次。")

    with c2:
        st.markdown("**HI热力图**")
        heat = build_hi_heatmap(hist_df)
        if len(heat) == 0:
            st.info("暂无足够数据生成HI热力图。")
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

            ax.set_title("HI 热力分布")
            plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

            flat = heat.stack(future_stack=True).dropna()
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
                "热力图说明：颜色越偏红，说明该日期该阀门的健康指数越低。"
                f"当前最低HI为 {worst_hi:.1f}，出现在 {worst_date} 的 {worst_valve}；"
                f"窗口内共有预警 {yellow_cnt} 条、高风险 {red_cnt} 条。"
            )
            st.markdown(
                "<div class='section-note'>热力图适合快速定位哪台阀门、哪几天出现了连续退化，是查找重点阀门最直观的入口。</div>",
                unsafe_allow_html=True,
            )
            if red_cnt > 0:
                st.warning("当前窗口内存在高风险记录，建议结合压力趋势与告警记录进行复盘。")

    with c3:
        st.markdown("**阀门HI对比**")
        comp = build_hi_compare(hist_df)
        if len(comp) == 0:
            st.info("暂无足够数据生成阀门HI对比。")
        else:
            fig, ax = plt.subplots(figsize=(4.8, 3.1))
            ax.bar(comp["valve_type"], comp["avg_HI"], color="#2e7d32")
            ax.set_ylim(0, 100)
            ax.set_ylabel("avg HI")
            ax.set_title("阀门 HI 对比")
            plt.xticks(rotation=20, ha="right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            comp_view = comp[["valve_type", "avg_HI", "min_HI", "red_days", "yellow_days"]].rename(
                columns={
                    "valve_type": "阀门",
                    "avg_HI": "平均HI",
                    "min_HI": "最低HI",
                    "red_days": "高风险天数",
                    "yellow_days": "预警天数",
                }
            )
            st.dataframe(comp_view, use_container_width=True, height=170)
            st.markdown(
                "<div class='section-note'>对比图用于识别长期偏弱阀门。平均HI偏低且红黄风险天数较多的阀门，应优先纳入巡检和复核计划。</div>",
                unsafe_allow_html=True,
            )

    st.info(build_leader_storyline(hist_df, hist_alerts))

    profile = build_valve_health_profile(hist_df, hist_alerts, valve_pick)
    if profile:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown(
                f"""
                <div class="evidence-panel">
                    <div class="hero-kicker">重点阀门健康档案</div>
                    <div class="hero-headline" style="font-size:1.15rem;">{profile['valve_type']}</div>
                    <div class="evidence-text">
                        最近记录：{profile['latest_date']}<br/>
                        当前状态：{profile['latest_risk']} / {profile['latest_stage']}<br/>
                        最新HI：{profile['latest_hi']:.1f}<br/>
                        AI观察天数：{profile['observe_days']}<br/>
                        AI升级/高风险天数：{profile['upgrade_days']}<br/>
                        告警数量：{profile['alert_count']}<br/>
                        最近告警状态：{profile['last_alert_status']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="evidence-panel">
                    <div class="hero-kicker">运行建议</div>
                    <div class="hero-headline" style="font-size:1.08rem;">{profile['action_suggestion']}</div>
                    <div class="hero-desc">{profile['risk_reason_path']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tab_ai(df_filtered: pd.DataFrame):
    st.subheader("AI预警中心")
    st.caption("这里集中展示多模型识别结果、共识过程和处置建议，便于技术核查与运行复盘。")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="ai_station")
    else:
        station_pick = STATION_SCOPE

    ai_df = _slice_by_station(df_filtered, station_pick)
    if len(ai_df) == 0:
        st.warning("当前范围暂无AI分析数据。")
        return

    valve_opts = sorted(ai_df["valve_type"].unique())
    valve_pick = st.selectbox("目标阀门", valve_opts, key="ai_valve")
    case_station = station_pick if station_pick != "全部站点" else str(
        ai_df[ai_df["valve_type"] == valve_pick].sort_values("consensus_score", ascending=False).iloc[0]["station"]
    )
    case_df = ai_df[ai_df["valve_type"] == valve_pick].copy()
    if station_pick == "全部站点":
        case_df = case_df[case_df["station"] == case_station].copy()
    case_replay = build_case_replay(ai_df if station_pick != "全部站点" else case_df, case_station, valve_pick)
    vote_matrix = build_model_vote_matrix(ai_df if station_pick != "全部站点" else case_df, case_station, valve_pick)
    local_summary = build_validation_summary(ai_df, pd.DataFrame())
    focus_row = case_df.sort_values(["consensus_score", "date"], ascending=[False, False]).iloc[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI观察记录数", int(ai_df["risk_stage"].eq("AI观察").sum()))
    m2.metric("AI升级记录数", int(ai_df["risk_stage"].eq("AI升级").sum()))
    m3.metric("AI高风险记录数", int(ai_df["risk_stage"].eq("AI高风险").sum()))
    m4.metric("IF/LOF可建模阀组", local_summary.get("eligible_group_count", 0))

    st.markdown(
        "<div class='section-note'>AI预警中心展示 Isolation Forest、局部离群因子、时序突变检测和退化趋势引擎的综合结果。首屏只给统一结论，这里则展开模型证据、投票过程和处置建议。</div>",
        unsafe_allow_html=True,
    )

    trend = (
        case_df.groupby("date")
        .agg(
            observe_cnt=("risk_stage", lambda s: (s == "AI观察").sum()),
            upgrade_cnt=("risk_stage", lambda s: (s == "AI升级").sum()),
            high_cnt=("risk_stage", lambda s: (s == "AI高风险").sum()),
        )
        .reset_index()
        .sort_values("date")
    )
    trend["date_dt"] = pd.to_datetime(trend["date"])
    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.plot(trend["date_dt"], trend["observe_cnt"], marker="o", label="AI观察")
    ax.plot(trend["date_dt"], trend["upgrade_cnt"], marker="o", label="AI升级")
    ax.plot(trend["date_dt"], trend["high_cnt"], marker="o", label="AI高风险")
    ax.set_title(f"{case_station} / {valve_pick} 异常时间线")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    score_cols = st.columns(5)
    score_cols[0].metric("IF分位", f"{float(pd.to_numeric(pd.Series([focus_row.get('ai_if_pct', np.nan)]), errors='coerce').iloc[0]) if pd.notna(focus_row.get('ai_if_pct', np.nan)) else 0:.1f}")
    score_cols[1].metric("LOF分位", f"{float(pd.to_numeric(pd.Series([focus_row.get('ai_lof_pct', np.nan)]), errors='coerce').iloc[0]) if pd.notna(focus_row.get('ai_lof_pct', np.nan)) else 0:.1f}")
    score_cols[2].metric("突变得分", f"{float(pd.to_numeric(pd.Series([focus_row.get('ai_shift_score', np.nan)]), errors='coerce').iloc[0]) if pd.notna(focus_row.get('ai_shift_score', np.nan)) else 0:.1f}")
    score_cols[3].metric("退化得分", f"{float(pd.to_numeric(pd.Series([focus_row.get('ai_degradation_score', np.nan)]), errors='coerce').iloc[0]) if pd.notna(focus_row.get('ai_degradation_score', np.nan)) else 0:.1f}")
    score_cols[4].metric("投票数 / 共识分", f"{int(focus_row.get('ai_vote_count', 0))} / {float(pd.to_numeric(pd.Series([focus_row.get('consensus_score', np.nan)]), errors='coerce').iloc[0]) if pd.notna(focus_row.get('consensus_score', np.nan)) else 0:.1f}")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="evidence-panel">
                <div class="hero-kicker">单阀案例回放</div>
                <div class="hero-headline" style="font-size:1.15rem;">{case_station} / {valve_pick}</div>
                <div class="evidence-text">
                    数据区间：{case_replay.get('date_start', '-')} 至 {case_replay.get('date_end', '-')}<br/>
                    重点日期：{case_replay.get('focus_date', '-')}<br/>
                    当前阶段：{case_replay.get('focus_stage', '-')}<br/>
                    原因链：{case_replay.get('focus_reason', '-')}<br/>
                    处置建议：{case_replay.get('action_suggestion', '-')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("**模型投票矩阵**")
        vote_view = vote_matrix.rename(
            columns={
                "date": "日期",
                "ai_if_score": "IF原始分",
                "ai_if_pct": "IF分位",
                "if_vote": "IF投票",
                "ai_lof_score": "LOF原始分",
                "ai_lof_pct": "LOF分位",
                "lof_vote": "LOF投票",
                "ai_shift_score": "突变得分",
                "shift_vote": "突变投票",
                "ai_degradation_score": "退化得分",
                "degradation_vote": "退化投票",
                "ai_vote_count": "投票数",
                "ai_confidence": "AI置信度",
                "consensus_score": "共识分",
                "risk_stage": "风险阶段",
            }
        )
        st.dataframe(vote_view, use_container_width=True, height=220)

        evidence_cols = [
            "date",
            "station",
            "valve_type",
            "Risk_final",
            "ai_if_pct",
            "ai_lof_pct",
            "ai_shift_score",
            "ai_degradation_score",
            "ai_vote_count",
            "ai_confidence",
            "consensus_score",
            "risk_stage",
            "risk_reason_path",
            "action_suggestion",
        ]
        st.markdown("**异常证据明细**")
        detail_view = case_df.sort_values("date", ascending=False)[evidence_cols].rename(
            columns={
                "date": "日期",
                "station": "站点",
                "valve_type": "阀门",
                "Risk_final": "规则风险",
                "ai_if_pct": "IF分位",
                "ai_lof_pct": "LOF分位",
                "ai_shift_score": "突变得分",
                "ai_degradation_score": "退化得分",
                "ai_vote_count": "投票数",
                "ai_confidence": "AI置信度",
                "consensus_score": "共识分",
                "risk_stage": "最终阶段",
                "risk_reason_path": "原因链",
                "action_suggestion": "处置建议",
            }
        )
        st.dataframe(detail_view, use_container_width=True, height=280)

    with right:
        quality_view = build_data_quality_table(ai_df)
        quality_view = quality_view.rename(
            columns={
                "station": "站点",
                "valve_type": "阀门",
                "samples": "样本数",
                "date_start": "起始日期",
                "date_end": "结束日期",
                "missing_rate": "缺失率(%)",
                "if_lof_ready": "IF/LOF可用",
                "shift_ready": "突变检测可用",
                "data_source_tag": "数据来源",
            }
        )
        st.markdown(
            "<div class='evidence-panel'><div class='hero-kicker'>数据质量状态</div><div class='hero-headline' style='font-size:1.15rem;'>模型可用性与样本准备情况</div><div class='hero-desc'>IF 和 LOF 需要同站点同阀门样本不少于30条，时序突变检测需要不少于14条；样本不足时系统自动降级为机理健康与退化趋势监测。</div></div>",
            unsafe_allow_html=True,
        )
        st.dataframe(quality_view, use_container_width=True, height=280)

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
            comp = comp.rename(
                columns={
                    "station": "站点",
                    "avg_HI": "平均HI",
                    "red_days": "高风险天数",
                    "yellow_days": "预警天数",
                    "activity": "活动频次",
                }
            )
            st.dataframe(comp, use_container_width=True)
        else:
            latest = dash_df.sort_values(["valve_type", "date"]).groupby("valve_type").tail(1)
            latest_view = latest[
                ["date", "station", "valve_type", "HI_final", "Risk_final", "risk_stage", "ai_confidence", "action_suggestion"]
            ].rename(
                columns={
                    "date": "日期",
                    "station": "站点",
                    "valve_type": "阀门",
                    "HI_final": "HI",
                    "Risk_final": "规则风险",
                    "risk_stage": "AI阶段",
                    "ai_confidence": "AI置信度",
                    "action_suggestion": "处置建议",
                }
            )
            st.dataframe(
                latest_view,
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

    focus_row = dash_df.sort_values(["consensus_score", "date"], ascending=[False, False]).iloc[0]
    focus_alerts = dash_alerts[dash_alerts["valve_type"] == focus_row["valve_type"]].copy()
    st.markdown(
        f"""
        <div class="section-note">
            重点阀门健康档案：{focus_row['valve_type']} | 当前风险 {focus_row['Risk_final']} / {focus_row['risk_stage']} |
            共识分 {float(focus_row.get('consensus_score', 0)):.1f} |
            最近告警状态 {str(focus_alerts.sort_values('date').iloc[-1]['status']) if len(focus_alerts) > 0 else '暂无告警'} |
            建议：{focus_row.get('action_suggestion', '当前运行平稳，建议按既定频次巡检。')}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    alert_view = view_alerts[show_cols].rename(
        columns={
            "id": "告警ID",
            "date": "日期",
            "station": "站点",
            "valve_type": "阀门",
            "risk_level": "风险等级",
            "trigger_source": "触发来源",
            "status": "当前状态",
            "owner": "责任人",
            "action_taken": "整改措施",
            "verification_result": "复验结果",
            "updated_at": "更新时间",
            "closed_at": "关闭时间",
        }
    )
    st.dataframe(alert_view, use_container_width=True)

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
    st.subheader("报表导出")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="report_station")
    else:
        station_pick = STATION_SCOPE

    rep_df = _slice_by_station(df_filtered, station_pick)
    rep_alerts = _slice_by_station(alerts_filtered, station_pick)

    if len(rep_df) == 0:
        st.warning("当前范围暂无可导出的数据。")
        return

    exp1, exp2, exp3 = st.columns([1, 1, 1], gap="small")

    with exp1:
        csv_data = rep_df.sort_values(["station", "valve_type", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "导出监测数据CSV",
            data=csv_data,
            file_name="psv_data_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp2:
        csv_alert = rep_alerts.sort_values(["station", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "导出告警数据CSV",
            data=csv_alert,
            file_name="psv_alerts_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with exp3:
        ai_rows = rep_df[rep_df["risk_stage"].isin(["AI观察", "AI升级", "AI高风险"])].copy()
        ai_csv = ai_rows[
            [
                "date",
                "station",
                "valve_type",
                "risk_stage",
                "ai_if_score",
                "ai_lof_score",
                "ai_shift_score",
                "ai_degradation_score",
                "ai_vote_count",
                "ai_confidence",
                "consensus_score",
                "risk_reason_path",
                "action_suggestion",
            ]
        ].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "导出AI异常明细CSV",
            data=ai_csv,
            file_name="psv_ai_anomalies.csv",
            mime="text/csv",
            use_container_width=True,
        )

    summary_text = generate_management_summary(rep_df, rep_alerts, station_pick, start_date, end_date, build_leader_storyline(rep_df, rep_alerts))
    st.text_area("管理摘要（可直接复制到周报或汇报材料）", value=summary_text, height=220)

    d1, d2 = st.columns(2, gap="small")
    with d1:
        st.download_button(
            "下载管理摘要TXT",
            data=summary_text.encode("utf-8"),
            file_name="management_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        quality_csv = build_data_quality_table(rep_df).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "下载数据质量清单CSV",
            data=quality_csv,
            file_name="psv_data_quality.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("**最近20条记录**")
    show_cols = [
        "date",
        "station",
        "valve_type",
        "data_source_tag",
        "p_now",
        "p_max",
        "level",
        "temp",
        "HI_final",
        "Risk_final",
        "risk_stage",
        "ai_confidence",
        "consensus_score",
        "action_suggestion",
    ]
    recent_view = rep_df.sort_values("date", ascending=False)[show_cols].head(20).rename(
        columns={
            "date": "日期",
            "station": "站点",
            "valve_type": "阀门",
            "data_source_tag": "数据来源",
            "p_now": "当前压力(MPa)",
            "p_max": "最高压力(MPa)",
            "level": "液位(%)",
            "temp": "温度(℃)",
            "HI_final": "HI",
            "Risk_final": "规则风险",
            "risk_stage": "AI阶段",
            "ai_confidence": "AI置信度",
            "consensus_score": "共识分",
            "action_suggestion": "处置建议",
        }
    )
    st.dataframe(recent_view, use_container_width=True)


# ================== Top Tabs ==================
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
