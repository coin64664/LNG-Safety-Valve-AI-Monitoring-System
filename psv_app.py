
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

from ai_engine import SKLEARN_OK, build_case_replay, build_validation_summary, run_scoring_pipeline
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
    build_patent_overview,
    generate_competition_brief,
    generate_management_summary,
    generate_patent_disclosure_outline,
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

    .hero-panel, .evidence-panel, .patent-panel {{
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
            "<h1 class='brand-title'>LNG储罐安全阀智能预警系统</h1>"
            "<div class='brand-subtitle'>规则机理 + 自适应历史基线 + 时序退化识别 + 异常共识升级</div>"
            "<div class='brand-chip'>专利核心版｜玉溪销售企业安全创新项目</div>"
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
st.set_page_config(page_title="LNG安全阀智能预警系统", layout="wide")

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
            <div style="font-size: 1.04rem; font-weight: 800; margin-bottom: 6px;">版本信息：v0.4 专利核心版</div>
            <div style="font-size: .95rem; line-height: 1.62;">
                更新内容：<br/>
                1. 新增“成果总览”，把项目亮点、专利创新点、数据质量和代表性实施例集中展示；<br/>
                2. AI引擎升级为“机理健康 + 自适应基线 + 时序退化 + 异常共识”四层方法链；<br/>
                3. 报表导出新增比赛摘要与专利交底书提纲，适配答辩与专利申报场景。
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
                    "consensus_score": float(row.get("consensus_score", np.nan)) if pd.notna(row.get("consensus_score", np.nan)) else None,
                    "ai_score_pct": float(row.get("ai_score_pct", np.nan)) if pd.notna(row.get("ai_score_pct", np.nan)) else None,
                    "risk_stage": row.get("risk_stage", "正常"),
                    "risk_reason_path": row.get("risk_reason_path", ""),
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
enable_ai = st.sidebar.checkbox("启用异常检测引擎（含 Isolation Forest）", value=True)
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


def render_signal_card(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">{title}</div>
            <div class="signal-value">{value}</div>
            <div class="signal-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab_overview(df_filtered: pd.DataFrame, alerts_filtered: pd.DataFrame):
    st.subheader("🏆 成果总览")
    storyline = build_leader_storyline(df_filtered, alerts_filtered)
    patent_df = build_patent_overview()
    source_mix = validation_summary.get("data_source_mix", {})
    source_text = " / ".join([f"{k}:{v}" for k, v in source_mix.items()]) if source_mix else "未标注"

    focus_station = str(df_filtered.sort_values("consensus_score", ascending=False).iloc[0]["station"])
    focus_valve = str(df_filtered.sort_values("consensus_score", ascending=False).iloc[0]["valve_type"])
    case = build_case_replay(df_filtered, focus_station, focus_valve)
    brief = generate_competition_brief(df_filtered, validation_summary, case)

    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Patent Core Method</div>
            <div class="hero-headline">一种基于自适应健康指数与分层异常共识的LNG储罐安全阀智能预警方法</div>
            <div class="hero-desc">这不是“单一算法演示”，而是一套围绕安全阀真实运维场景构建的工业智能预警方法：先做机理健康评估，再做历史基线偏离，再识别时序退化，最后通过异常共识完成风险升级。</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        render_signal_card("一句话创新结论", "规则 + AI + 共识", "把机理规则、历史基线和异常持续性串成一个能申请专利的方法链。")
    with c2:
        render_signal_card("当前最值得关注阀门", focus_valve, case.get("focus_reason", "暂无重点案例"))
    with c3:
        render_signal_card("AI提前识别/升级", f"{validation_summary.get('observe_count', 0)} / {validation_summary.get('escalate_count', 0) + validation_summary.get('high_count', 0)}", "前者表示进入观察，后者表示进入升级或高风险。")
    with c4:
        render_signal_card("闭环与数据状态", f"{validation_summary.get('close_rate', 0.0):.1f}%", f"数据来源：{source_text}")

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="evidence-panel">
                <div class="hero-kicker">Competition Brief</div>
                <div class="hero-headline" style="font-size:1.18rem;">项目摘要与代表性实施例</div>
                <div class="evidence-text">{brief.replace(chr(10), '<br/>')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='section-note'>{storyline}</div>", unsafe_allow_html=True)
        st.dataframe(quality_table, use_container_width=True, height=220)

    with right:
        st.markdown(
            """
            <div class="patent-panel">
                <div class="hero-kicker">Patent Scope</div>
                <div class="hero-headline" style="font-size:1.18rem;">专利创新点总览</div>
                <div class="hero-desc">保护点不落在“孤立森林本身”，而落在面向LNG安全阀场景的组合方法。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(patent_df, use_container_width=True, height=220)


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
            st.markdown(
                "<div class='section-note'>专利视角：这张图支撑“机理健康层 + 时序退化层”。评委能直接看到压力接近整定值、连续上升和异常出现的先后关系。</div>",
                unsafe_allow_html=True,
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
            st.markdown(
                "<div class='section-note'>专利视角：这张图支撑“分层风险升级方法”。它把单天点状异常拉成时间-阀门二维证据，说明系统不仅会报异常，还能识别持续性退化。</div>",
                unsafe_allow_html=True,
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
            st.markdown(
                "<div class='section-note'>专利视角：这张对比图支撑“同站多阀优先级排序”，说明系统不只是判风险，还能告诉管理者应优先关注哪一只阀门。</div>",
                unsafe_allow_html=True,
            )

    st.info(build_leader_storyline(hist_df, hist_alerts))


def render_tab_ai(df_filtered: pd.DataFrame):
    st.subheader("🤖 AI预警中心")
    st.caption("方法链：机理健康层 → 自适应基线层 → 时序退化层 → 异常共识层")

    if IS_LEADER:
        station_pick = st.selectbox("站点", ["全部站点"] + sorted(df_filtered["station"].unique()), key="ai_station")
    else:
        station_pick = STATION_SCOPE

    ai_df = _slice_by_station(df_filtered, station_pick)
    if len(ai_df) == 0:
        st.warning("该范围内暂无数据。")
        return

    valve_opts = sorted(ai_df["valve_type"].unique())
    valve_pick = st.selectbox("案例阀门", valve_opts, key="ai_valve")
    case_station = station_pick if station_pick != "全部站点" else str(
        ai_df[ai_df["valve_type"] == valve_pick].sort_values("consensus_score", ascending=False).iloc[0]["station"]
    )
    case_df = ai_df[ai_df["valve_type"] == valve_pick].copy()
    if station_pick == "全部站点":
        case_df = case_df[case_df["station"] == case_station].copy()
    case_replay = build_case_replay(ai_df if station_pick != "全部站点" else case_df, case_station, valve_pick)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("AI观察数", int(ai_df["risk_stage"].eq("AI观察").sum()))
    m2.metric("AI升级数", int(ai_df["risk_stage"].eq("AI升级").sum()))
    m3.metric("AI高风险数", int(ai_df["risk_stage"].eq("AI高风险").sum()))
    m4.metric("满足AI样本组", validation_summary.get("eligible_group_count", 0))

    st.markdown(
        "<div class='section-note'>当前AI不再只依赖 Isolation Forest 单点判断，而是将历史基线偏离、连续退化、规则风险和异常持续性综合为“异常共识”。</div>",
        unsafe_allow_html=True,
    )

    trend = (
        ai_df.groupby("date")
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
    ax.set_title("异常共识时间线")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=30)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="evidence-panel">
                <div class="hero-kicker">Case Replay</div>
                <div class="hero-headline" style="font-size:1.15rem;">{case_station} / {valve_pick}</div>
                <div class="evidence-text">
                    样本区间：{case_replay.get('date_start', '—')} 至 {case_replay.get('date_end', '—')}<br/>
                    重点日期：{case_replay.get('focus_date', '—')}<br/>
                    风险阶段：{case_replay.get('focus_stage', '—')}<br/>
                    原因链：{case_replay.get('focus_reason', '—')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        evidence_cols = [
            "date",
            "station",
            "valve_type",
            "Risk",
            "ai_score_pct",
            "baseline_dev_score",
            "degradation_score",
            "consensus_score",
            "risk_stage",
            "risk_reason_path",
        ]
        st.dataframe(case_df.sort_values("date", ascending=False)[evidence_cols], use_container_width=True, height=280)

    with right:
        quality_view = build_data_quality_table(ai_df)
        st.markdown(
            "<div class='patent-panel'><div class='hero-kicker'>Data Quality</div><div class='hero-headline' style='font-size:1.15rem;'>AI建模条件与样本质量</div><div class='hero-desc'>样本不足的阀门组不参与AI升级，只保留规则风险。这一点会直接体现在比赛答辩和专利实施例里。</div></div>",
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

    focus = rep_df.sort_values("consensus_score", ascending=False).iloc[0]
    case_replay = build_case_replay(rep_df, str(focus["station"]), str(focus["valve_type"]))
    summary_text = generate_management_summary(rep_df, rep_alerts, station_pick, start_date, end_date, build_leader_storyline(rep_df, rep_alerts))
    competition_text = generate_competition_brief(rep_df, build_validation_summary(rep_df, rep_alerts), case_replay)
    patent_text = generate_patent_disclosure_outline(rep_df, [case_replay])

    st.text_area("管理摘要（可直接贴PPT）", value=summary_text, height=220)
    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        st.download_button(
            "下载管理摘要TXT",
            data=summary_text.encode("utf-8"),
            file_name="management_summary.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "下载比赛摘要TXT",
            data=competition_text.encode("utf-8"),
            file_name="competition_brief.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with d3:
        st.download_button(
            "下载专利交底书草案MD",
            data=patent_text.encode("utf-8"),
            file_name="patent_disclosure_outline.md",
            mime="text/markdown",
            use_container_width=True,
        )

    cc1, cc2 = st.columns(2, gap="large")
    with cc1:
        st.text_area("比赛摘要版", value=competition_text, height=220)
    with cc2:
        st.text_area("专利交底书提纲", value=patent_text, height=220)

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
        "consensus_score",
        "ai_score_pct",
        "ai_observe_flag",
        "ai_escalate_flag",
    ]
    st.dataframe(rep_df.sort_values("date", ascending=False)[show_cols].head(20), use_container_width=True)


# ================== Top Tabs ==================
# 默认首开第一个Tab：成果总览
overview_tab, hist_tab, ai_tab, dash_tab, alert_tab, report_tab = st.tabs(
    ["成果总览", "历史分析", "AI预警中心", "驾驶舱", "告警闭环", "报表导出"]
)

with overview_tab:
    render_tab_overview(df_f, alerts_f)

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
