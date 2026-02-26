
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib as mpl
from matplotlib import font_manager

# ================== 数据存储：Supabase（多人共享） ==================
# 说明：不改变你项目任何业务逻辑，只把“本地CSV”替换为“Supabase云数据库”。
# 建议把密钥放在 Streamlit Secrets 或环境变量里，避免写进代码。

try:
    from supabase import create_client
    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False

# 优先从 st.secrets 读取，其次从环境变量读取
SUPABASE_URL = None
SUPABASE_KEY = None
try:
    SUPABASE_URL = st.secrets.get("SUPABASE_URL", None)
    SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", None)
except Exception:
    pass

SUPABASE_URL = SUPABASE_URL or os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = SUPABASE_KEY or os.getenv("SUPABASE_KEY", "")

# 你可以把 USE_SUPABASE 设为 True 强制使用云端；如果密钥缺失会自动回退本地CSV（便于你本地调试）
USE_SUPABASE = True
supabase = None
if USE_SUPABASE and SUPABASE_OK and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
elif USE_SUPABASE:
    # 密钥缺失时提示一次，但仍允许回退本地CSV以免程序直接挂掉
    st.sidebar.warning("⚠️ 未检测到 Supabase 配置（SUPABASE_URL / SUPABASE_KEY），将回退为本地CSV存储。")
    USE_SUPABASE = False

# =====================================================================


# ================== 机器学习：Isolation Forest（无监督异常检测） ==================
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ================== Matplotlib 中文字体修复（避免图表标题/标签乱码） ==================
def _setup_cjk_font():
    # 优先使用系统中常见的中文字体；在多数 Linux 环境下 NotoSansCJK 可用
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

    # 兜底：直接指定 NotoSansCJK 字体文件（容器里通常有）
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    if os.path.exists(font_path):
        fp = font_manager.FontProperties(fname=font_path)
        mpl.rcParams["font.family"] = fp.get_name()
        mpl.rcParams["axes.unicode_minus"] = False

_setup_cjk_font()

# ================== 配置区 ==================
SET_P = 1.32  # 安全阀整定压力（MPa）
DATA_FILE = "psv_data.csv"
APP_PASSWORD = "adsf0608"  # 简单口令（可后续换成账号体系）

st.set_page_config(page_title="LNG安全阀健康监测系统", layout="wide")

# ================== 登录 ==================
st.sidebar.title("🔐 访问控制")
user_password = st.sidebar.text_input("请输入密码", type="password")
if user_password != APP_PASSWORD:
    st.warning("请输入正确密码后进入系统 当前版本v0.2 开发：YXY。")
    st.stop()


# ================== AI 设置（无监督异常检测） ==================
st.sidebar.divider()
st.sidebar.header("🧠 AI 异常检测（Isolation Forest）")
enable_ai = st.sidebar.checkbox("启用 AI 异常检测", value=True)
contamination = st.sidebar.slider("异常比例（越大越敏感）", min_value=0.02, max_value=0.20, value=0.08, step=0.01)

if enable_ai and not SKLEARN_OK:
    st.sidebar.warning("当前环境缺少 scikit-learn，AI 异常检测不可用。可执行：pip install scikit-learn")
    enable_ai = False

# ================== 标题 ==================
st.title("玉溪销售加气站 LNG 安全阀 AI 健康监测与异常识别系统")
st.caption("基于每日人工上报数据的风险预警、趋势分析与无监督异常检测（Isolation Forest）（整定压力：1.32 MPa）")

# ================== 初始化数据文件/云端表 ==================
# 你原先用 CSV 存储；这里保持 CSV 逻辑不删，只是在 USE_SUPABASE=True 且配置齐全时改用 Supabase 表：psv_data
if not USE_SUPABASE:
    if not os.path.exists(DATA_FILE):
        df_init = pd.DataFrame(
            columns=["date", "valve_type", "p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]
        )
        df_init.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

def load_data() -> pd.DataFrame:
    # ---- 云端：Supabase ----
    if USE_SUPABASE and supabase is not None:
        resp = supabase.table("psv_data").select("*").execute()
        data = resp.data or []
        df0 = pd.DataFrame(data)
        if len(df0) == 0:
            return df0
        # Supabase 返回的 date 可能是字符串
        df0["date"] = pd.to_datetime(df0["date"]).dt.date
        # 兜底：防止字符串/空值
        for col in ["p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]:
            if col in df0.columns:
                df0[col] = pd.to_numeric(df0[col], errors="coerce")
        df0 = df0.dropna(subset=["date", "valve_type", "p_max"])
        return _normalize_df(df0)

    # ---- 本地：CSV（回退）----
    df0 = pd.read_csv(DATA_FILE)
    if len(df0) == 0:
        return df0
    df0["date"] = pd.to_datetime(df0["date"]).dt.date
    # 兜底：防止字符串/空值
    for col in ["p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]:
        if col in df0.columns:
            df0[col] = pd.to_numeric(df0[col], errors="coerce")
    df0 = df0.dropna(subset=["date", "valve_type", "p_max"])
    return _normalize_df(df0)




def _normalize_df(df0: pd.DataFrame) -> pd.DataFrame:
    """统一做数据清洗/约束，避免录入或存储导致的图表异常。"""
    if df0 is None or len(df0) == 0:
        return df0

    df = df0.copy()

    # 类型兜底
    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in ["p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 合理范围（防止误录）
    if "p_now" in df.columns:
        df["p_now"] = df["p_now"].clip(lower=0, upper=2)
    if "p_max" in df.columns:
        df["p_max"] = df["p_max"].clip(lower=0, upper=2)
    if "level" in df.columns:
        df["level"] = df["level"].clip(lower=0, upper=100)
    if "temp" in df.columns:
        df["temp"] = df["temp"].clip(lower=-50, upper=80)

    # 物理约束：当日最高压力 >= 当前压力（若违反，按 p_now 修正 p_max）
    if "p_now" in df.columns and "p_max" in df.columns:
        m = df["p_max"].notna() & df["p_now"].notna() & (df["p_max"] < df["p_now"])
        if m.any():
            df.loc[m, "p_max"] = df.loc[m, "p_now"]

    # 同一阀门同一天重复录入：保留最后一条（避免图表“看起来不对”）
    if set(["date", "valve_type"]).issubset(df.columns):
        df = df.sort_values(["valve_type", "date"]).drop_duplicates(
            subset=["date", "valve_type"], keep="last"
        )

    df = df.dropna(subset=["date", "valve_type", "p_max"])
    return df
def compute_scores(df0: pd.DataFrame, enable_ai: bool, contamination: float) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return df0

    df0 = _normalize_df(df0)
    if df0 is None or len(df0) == 0:
        return df0

    df = df0.copy()
    df = df.sort_values(["valve_type", "date"]).reset_index(drop=True)

    # ratio：当日最高压力接近整定压力的程度
    df["ratio"] = df["p_max"] / SET_P

    # slope：3日压力变化趋势（按阀门分组）
    df["slope"] = (
        df.groupby("valve_type")["p_max"]
        .apply(lambda s: s.diff().rolling(3).mean())
        .reset_index(level=0, drop=True)
    )

    HI = np.full(len(df), 100.0)

    # A) 接近整定压力扣分（分段）
    HI -= np.where(df["ratio"] >= 1.00, 35, 0)
    HI -= np.where((df["ratio"] >= 0.98) & (df["ratio"] < 1.00), 20, 0)
    HI -= np.where((df["ratio"] >= 0.95) & (df["ratio"] < 0.98), 10, 0)

    # B) 连续上升趋势扣分（3日均值）
    HI -= np.where(df["slope"] > 0.01, 10, 0)   # 3天平均每天 +0.01MPa
    HI -= np.where(df["slope"] > 0.02, 10, 0)   # 更陡再扣一次

    # C) 动作/微放散扣分（维护触发信号）
    HI -= df.get("psv_act", 0).fillna(0) * 30
    HI -= df.get("psv_weeping", 0).fillna(0) * 15

    # D) 高温 + 高液位 + 高压力（风险叠加因子）
    HI -= np.where(
        (df.get("temp", 0).fillna(0) >= 33)
        & (df.get("level", 0).fillna(0) >= 80)
        & (df["ratio"] >= 0.95),
        10,
        0,
    )

    df["HI"] = np.clip(HI, 0, 100)

    def risk(x: float) -> str:
        if x >= 85:
            return "🟢 安全"
        if x >= 70:
            return "🟡 预警"
        return "🔴 高风险"

    df["Risk"] = df["HI"].apply(risk)
    df["Activity"] = df.get("psv_act", 0).fillna(0) + df.get("psv_weeping", 0).fillna(0)



    # ================== AI 异常检测（Isolation Forest） ==================
    # 说明：无监督算法，不需要故障标签；用于发现“模式异常”的运行日，补足规则阈值的盲区。
    df["AI_anomaly"] = False
    df["AI_score"] = np.nan

    if enable_ai and SKLEARN_OK:
        features = ["p_now", "p_max", "level", "temp", "ratio", "slope", "Activity"]
        for valve, g in df.groupby("valve_type"):
            idx = g.index
            # 数据太少时不做AI（避免误报）
            if len(g) < 10:
                continue

            X = g[features].copy()
            # 缺失值用该阀门的中位数填充
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median(numeric_only=True))

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X.values)

            iso = IsolationForest(
                n_estimators=200,
                contamination=float(contamination),
                random_state=42,
            )
            iso.fit(Xs)

            pred = iso.predict(Xs)  # -1=异常, 1=正常
            score = -iso.score_samples(Xs)  # 值越大越“异常”

            df.loc[idx, "AI_anomaly"] = (pred == -1)
            df.loc[idx, "AI_score"] = score

        # 将AI异常作为“额外风险因子”融合到健康指数中（轻量融合，避免过度影响）
        df["HI_final"] = np.clip(df["HI"] - df["AI_anomaly"].astype(int) * 10, 0, 100)
    else:
        df["HI_final"] = df["HI"]

    df["Risk_final"] = df["HI_final"].apply(risk)

    return df

df_raw = load_data()

# ================== 侧边栏：录入 ==================
st.sidebar.divider()
st.sidebar.header("📝 每日数据录入")

valve_type = st.sidebar.selectbox("选择安全阀类型", ["泵后安全阀", "储罐主阀", "储罐辅阀"])
date = st.sidebar.date_input("日期")
p_now = st.sidebar.number_input("当前压力 p_now (MPa)", 0.0, 2.0, 1.20, 0.01)
p_max = st.sidebar.number_input("当日最高压力 p_max (MPa)", 0.0, 2.0, 1.20, 0.01, help="建议：p_max ≥ p_now；若输入小于 p_now，系统会自动按 p_now 修正。")
level = st.sidebar.number_input("液位 level (%)", 0, 100, 60)
temp = st.sidebar.number_input("环境温度 temp (℃)", -30, 60, 25)
psv_act = st.sidebar.selectbox("是否动作", ["否", "是"])
psv_weeping = st.sidebar.selectbox("是否微放散/嘶嘶声", ["否", "是"])

if st.sidebar.button("保存并计算", use_container_width=True):
    # 数据校验：当日最高压力应 >= 当前压力
    p_now_f = float(p_now)
    p_max_f = float(p_max)
    if p_max_f < p_now_f:
        st.sidebar.warning(f"已自动修正：p_max({p_max_f:.2f}) < p_now({p_now_f:.2f})，将 p_max 设为 {p_now_f:.2f}")
        p_max_f = p_now_f

    # 你原来的录入字段与逻辑保持不变，只替换“保存位置”
    if USE_SUPABASE and supabase is not None:
        supabase.table("psv_data").upsert(
            {
                "date": str(date),
                "valve_type": valve_type,
                "p_now": p_now_f,
                "p_max": p_max_f,
                "level": int(level),
                "temp": int(temp),
                "psv_act": 1 if psv_act == "是" else 0,
                "psv_weeping": 1 if psv_weeping == "是" else 0,
            },
            on_conflict="date,valve_type",
        ).execute()
        st.sidebar.success("✅ 数据已保存到 Supabase（云端）")
        st.rerun()
    else:
        new_row = pd.DataFrame(
            [{
                "date": date,
                "valve_type": valve_type,
                "p_now": p_now_f,
                "p_max": p_max_f,
                "level": level,
                "temp": temp,
                "psv_act": 1 if psv_act == "是" else 0,
                "psv_weeping": 1 if psv_weeping == "是" else 0,
            }]
        )

        df_to_save = pd.concat([df_raw, new_row], ignore_index=True)
        # 同一阀门同一天重复录入：保留最后一条
        df_to_save = df_to_save.sort_values(['valve_type','date']).drop_duplicates(subset=['date','valve_type'], keep='last')
        df_to_save.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
        st.sidebar.success("✅ 数据已保存（刷新页面可看到更新）")

# 重新加载 + 计算
df_raw = load_data()
df = compute_scores(df_raw, enable_ai=enable_ai, contamination=contamination)

# ================== 主页面 ==================
if len(df) == 0:
    st.info("当前还没有数据：请在左侧录入并点击【保存并计算】。")
    st.stop()

# 日期范围筛选（领导展示会更像“系统”）
min_d, max_d = df["date"].min(), df["date"].max()
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    start_date = st.date_input("开始日期", value=min_d, min_value=min_d, max_value=max_d, key="start_date")
with colB:
    end_date = st.date_input("结束日期", value=max_d, min_value=min_d, max_value=max_d, key="end_date")
with colC:
    st.caption("提示：如果数据量小，建议先录入 10–30 天")

df_f = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
if len(df_f) == 0:
    st.warning("所选日期范围内没有数据。")
    st.stop()

with st.expander("📥 导出数据（所选日期范围）", expanded=False):
    csv_bytes = df_f.sort_values(["valve_type", "date"]).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="下载 CSV",
        data=csv_bytes,
        file_name="psv_data_filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============ 总览看板 ============
st.subheader("📌 总览看板（最新状态）")

latest_by_valve = df.sort_values(["valve_type", "date"]).groupby("valve_type").tail(1)
cols = st.columns(3)
for i, valve in enumerate(["泵后安全阀", "储罐主阀", "储罐辅阀"]):
    block = latest_by_valve[latest_by_valve["valve_type"] == valve]
    if len(block) == 0:
        cols[i].metric(f"{valve} 当前健康指数", "—")
        cols[i].metric(f"{valve} 风险等级", "—")
        continue
    row = block.iloc[0]
    cols[i].metric(f"{valve} 当前健康指数", f"{row['HI_final']:.1f}")
    cols[i].metric(f"{valve} 风险等级", row["Risk_final"])
    cols[i].caption(f"最高压力占整定比例：{row['ratio'] * 100:.1f}% ｜ 动作:{int(row.get('psv_act',0))} 微放散:{int(row.get('psv_weeping',0))}")

st.divider()

# ============ 详情：按阀门趋势 ============
st.subheader("📈 单阀趋势（压力 & 健康指数）")
valve_pick = st.selectbox("选择查看的阀门", sorted(df_f["valve_type"].unique()), index=0)

vdf = df_f[df_f["valve_type"] == valve_pick].sort_values("date").copy()
vdf["date_dt"] = pd.to_datetime(vdf["date"])

c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots()
    ax.plot(vdf["date_dt"], vdf["p_max"], marker="o", label="p_max")
    if "p_now" in vdf.columns:
        ax.plot(vdf["date_dt"], vdf["p_now"], marker="o", linestyle="--", label="p_now")
    ax.axhline(SET_P, linestyle="--", label="整定压力 1.32MPa")
    ax.set_title(f"{valve_pick}：当日最高压力趋势")
    ax.set_ylabel("MPa")
    ax.set_xlabel("日期")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=30)
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    ax.plot(vdf["date_dt"], vdf["HI_final"], marker="o")
    ax.set_title(f"{valve_pick}：健康指数趋势（HI，AI融合）")
    ax.set_ylabel("HI (0-100)")
    ax.set_xlabel("日期")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=30)
    st.pyplot(fig)

st.divider()

# ============ 高级可视化 ============
st.subheader("🧠 高级可视化")
st.caption("建议阅读顺序：①热力图找“哪天哪阀变差” → ②对比图决定“优先处理哪只阀” → ③相关图解释“压力接近整定是否更容易动作/微放散”。")

g1, g2, g3 = st.columns(3, gap="small")

# ---- 1) 热力图：健康随时间（小图）----
with g1:
    st.markdown("**① 热力图：健康随时间**")
    st.caption("颜色越深（偏紫）代表 HI 越低。")

    heat = df_f.pivot_table(index="valve_type", columns="date", values="HI_final", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_title("HI 热力图")

    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(list(heat.index))

    # 日期太多时做抽样，避免小图挤爆
    cols = list(heat.columns)
    if len(cols) <= 10:
        tick_idx = list(range(len(cols)))
    else:
        tick_idx = sorted(set(np.linspace(0, len(cols) - 1, 8).round().astype(int).tolist()))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([pd.to_datetime(cols[i]).strftime("%m-%d") for i in tick_idx], rotation=45, ha="right")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="HI")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---- 2) 条形图：阀门对比（小图）----
with g2:
    st.markdown("**② 对比：平均HI & 预警天数**")
    st.caption("平均HI越低、红/黄天数越多 → 越优先处理。")

    summary = (
        df_f.groupby("valve_type")
        .agg(
            avg_HI=("HI_final", "mean"),
            min_HI=("HI_final", "min"),
            red_days=("Risk_final", lambda s: (s == "🔴 高风险").sum()),
            yellow_days=("Risk_final", lambda s: (s == "🟡 预警").sum()),
            act_cnt=("psv_act", "sum"),
            weep_cnt=("psv_weeping", "sum"),
        )
        .reset_index()
        .sort_values("avg_HI", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(summary["valve_type"], summary["avg_HI"])
    ax.set_title("平均HI（越高越好）")
    ax.set_ylabel("avg HI")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # 小图下方给一行“结论提示”，领导更容易看懂
    worst = summary.sort_values("avg_HI").head(1).iloc[0]
    st.info(f"优先关注：{worst['valve_type']}（平均HI≈{worst['avg_HI']:.1f}，高风险天数={int(worst['red_days'])}，预警天数={int(worst['yellow_days'])}）")

# ---- 3) 散点图：压力 vs 活动（小图）----
with g3:
    st.markdown("**③ 相关：压力 vs 动作/微放散**")
    st.caption("点越靠上代表动作/微放散越多；用于验证阈值设置是否合理。")

    sdf = df_f.copy()
    jitter = (np.random.default_rng(0).random(len(sdf)) - 0.5) * 0.06
    y = sdf["Activity"].values + jitter

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(sdf["p_max"], y)
    ax.set_title("p_max vs 活动")
    ax.set_xlabel("p_max (MPa)")
    ax.set_ylabel("活动(0/1/2)")
    ax.set_yticks([0, 1, 2])
    ax.set_ylim(-0.3, 2.3)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    if sdf["p_max"].nunique() > 1 and sdf["Activity"].nunique() > 1:
        corr = np.corrcoef(sdf["p_max"], sdf["Activity"])[0, 1]
        st.metric("相关系数", f"{corr:.2f}")
        if "AI_anomaly" in sdf.columns:
            st.metric("AI 异常天数", int(sdf["AI_anomaly"].sum()))
    else:
        st.info("数据变化不足，暂无法计算相关性。建议多录入一些天数。")

st.divider()

# ============ 历史记录 ============
st.subheader("🗂 历史记录（最近 20 条）")
show_cols = ["date","valve_type","p_now","p_max","level","temp","psv_act","psv_weeping","HI_final","Risk_final","AI_anomaly","AI_score"]
if not set(show_cols).issubset(df.columns):
    st.dataframe(df.sort_values("date", ascending=False).head(20))
else:
    st.dataframe(df.sort_values("date", ascending=False)[show_cols].head(20))
