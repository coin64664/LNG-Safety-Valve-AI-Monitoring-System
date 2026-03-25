from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["station", "valve_type", "samples", "date_start", "date_end", "missing_rate", "if_lof_ready", "shift_ready", "data_source_tag"])

    work = df.copy()
    numeric_cols = ["p_now", "p_max", "level", "temp"]
    missing = work[numeric_cols].isna().mean(axis=1)
    work["row_missing_rate"] = missing
    quality = (
        work.groupby(["station", "valve_type"])
        .agg(
            samples=("date", "count"),
            date_start=("date", "min"),
            date_end=("date", "max"),
            missing_rate=("row_missing_rate", "mean"),
            if_lof_ready=("date", lambda s: len(s) >= 30),
            shift_ready=("date", lambda s: len(s) >= 14),
            data_source_tag=("data_source_tag", lambda s: " / ".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
        )
        .reset_index()
    )
    quality["missing_rate"] = (quality["missing_rate"] * 100).round(1)
    return quality


def _top_reason_text(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or "ai_reason_top1" not in df.columns:
        return "暂无"
    values = (
        df.loc[df["ai_reason_top1"].astype(str) != "-", "ai_reason_top1"]
        .value_counts()
        .head(3)
        .to_dict()
    )
    return "；".join([f"{k} {v}次" for k, v in values.items()]) if values else "暂无"


def generate_management_summary(df: pd.DataFrame, alerts_df: pd.DataFrame, station_pick: str, date_start, date_end, storyline: str) -> str:
    if df is None or len(df) == 0:
        return "当前范围暂无可导出数据。"

    avg_hi = float(pd.to_numeric(df["HI_final"], errors="coerce").mean())
    red_cnt = int(df["Risk_final"].eq("🔴 高风险").sum())
    warn_cnt = int(df["Risk_final"].eq("🟡 预警").sum())
    observe_cnt = int(df["risk_stage"].eq("AI观察").sum())
    upgrade_cnt = int(df["risk_stage"].isin(["AI升级", "AI高风险"]).sum())
    close_rate = float((alerts_df["status"] == "已关闭").mean() * 100) if alerts_df is not None and len(alerts_df) else 0.0
    return "\n".join(
        [
            "LNG储罐安全阀健康监测与风险预警系统 管理摘要",
            f"统计范围：{station_pick}",
            f"时间范围：{date_start} 至 {date_end}",
            f"平均健康指数：{avg_hi:.1f}",
            f"高风险记录数：{red_cnt}",
            f"预警记录数：{warn_cnt}",
            f"AI观察记录数：{observe_cnt}",
            f"AI升级/高风险记录数：{upgrade_cnt}",
            f"主要异常原因：{_top_reason_text(df)}",
            f"告警闭环率：{close_rate:.1f}%",
            storyline,
            "管理建议：优先复盘连续异常且接近整定压力的阀门，结合处置结果安排复检与巡检频次调整。",
        ]
    )


def generate_technical_paper_data(df: pd.DataFrame, alerts_df: pd.DataFrame, case_replay: dict, quality_table: pd.DataFrame | None = None) -> dict:
    if quality_table is None:
        quality_table = build_data_quality_table(df)

    total_rows = int(len(df)) if df is not None else 0
    stations = sorted(df["station"].dropna().astype(str).unique().tolist()) if df is not None and len(df) else []
    valves = sorted(df["valve_type"].dropna().astype(str).unique().tolist()) if df is not None and len(df) else []
    avg_hi = float(pd.to_numeric(df["HI_final"], errors="coerce").mean()) if df is not None and len(df) else 0.0
    high_cnt = int(df["risk_stage"].eq("AI高风险").sum()) if df is not None and len(df) else 0
    upgrade_cnt = int(df["risk_stage"].eq("AI升级").sum()) if df is not None and len(df) else 0
    observe_cnt = int(df["risk_stage"].eq("AI观察").sum()) if df is not None and len(df) else 0
    close_rate = float((alerts_df["status"] == "已关闭").mean() * 100) if alerts_df is not None and len(alerts_df) else 0.0

    return {
        "title": "基于多模型异常识别的LNG储罐安全阀健康监测与风险预警系统",
        "background": "针对 LNG 储罐安全阀运行中存在的连续退化难发现、单一阈值法提前量不足、异常原因难解释等问题，构建面向站点实际工况的智能预警系统。",
        "problem": "传统方法多依赖单点阈值或人工经验，难以兼顾全局异常、局部异常、时序突变和持续退化过程。",
        "architecture": "系统采用多站点分角色管理架构，数据层接入运行参数、告警闭环和审计记录，算法层由机理健康、历史基线、时序突变、多模型共识组成，展示层包含历史分析、AI预警、驾驶舱和闭环管理。",
        "method": "算法采用机理健康评分、Isolation Forest、Local Outlier Factor、时序突变检测和退化趋势引擎四路协同识别，通过投票与共识分实现分层风险升级。",
        "case": case_replay,
        "metrics": {
            "records": total_rows,
            "stations": stations,
            "valves": valves,
            "avg_hi": round(avg_hi, 1),
            "observe_count": observe_cnt,
            "upgrade_count": upgrade_cnt,
            "high_count": high_cnt,
            "close_rate": round(close_rate, 1),
        },
        "quality_table": quality_table.copy(),
        "value": "系统可将分散的运行记录转化为可追溯的风险识别结果和处置建议，支撑现场巡检优化、重点阀门优先级排序和管理闭环跟踪。",
        "impact": "该系统具备在 LNG 场站安全管理、设备健康监测和风险预警场景中的推广潜力。",
    }


def _pick_cjk_font():
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Micro Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name, font_manager.FontProperties(family=name)

    fallback_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ]
    for path in fallback_paths:
        if Path(path).exists():
            return font_manager.FontProperties(fname=path).get_name(), font_manager.FontProperties(fname=path)

    return "DejaVu Sans", font_manager.FontProperties(family="DejaVu Sans")


def _add_text_page(pdf: PdfPages, title: str, lines: Iterable[str], font_prop):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=18, fontweight="bold", fontproperties=font_prop, color="#173b63")
    y = 0.90
    for line in lines:
        fig.text(0.08, y, line, fontsize=11, fontproperties=font_prop, color="#1c2b42", va="top")
        y -= 0.045
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_technical_paper_pdf(paper_data: dict) -> bytes:
    buffer = BytesIO()
    font_name, font_prop = _pick_cjk_font()

    with PdfPages(buffer) as pdf:
        metrics = paper_data.get("metrics", {})
        case = paper_data.get("case", {})
        quality = paper_data.get("quality_table", pd.DataFrame())

        _add_text_page(
            pdf,
            paper_data.get("title", "技术论文"),
            [
                "一、项目背景",
                paper_data.get("background", ""),
                "二、场景问题",
                paper_data.get("problem", ""),
                "三、系统架构",
                paper_data.get("architecture", ""),
                "四、多模型集成预警方法",
                paper_data.get("method", ""),
            ],
            font_prop,
        )

        _add_text_page(
            pdf,
            "关键实施案例与结果",
            [
                f"记录总数：{metrics.get('records', 0)}",
                f"站点范围：{'、'.join(metrics.get('stations', [])) or '暂无'}",
                f"阀门类型：{'、'.join(metrics.get('valves', [])) or '暂无'}",
                f"平均健康指数：{metrics.get('avg_hi', 0)}",
                f"AI观察次数：{metrics.get('observe_count', 0)}",
                f"AI升级次数：{metrics.get('upgrade_count', 0)}",
                f"AI高风险次数：{metrics.get('high_count', 0)}",
                f"告警闭环率：{metrics.get('close_rate', 0)}%",
                "",
                f"代表阀门：{case.get('station', '-')} / {case.get('valve_type', '-')}",
                f"案例时间：{case.get('date_start', '-')} 至 {case.get('date_end', '-')}",
                f"关键日期：{case.get('focus_date', '-')}",
                f"风险阶段：{case.get('focus_stage', '-')}",
                f"原因链：{case.get('focus_reason', '-')}",
                f"处置建议：{case.get('action_suggestion', '-')}",
                "",
                "五、结果与价值",
                paper_data.get('value', ''),
                "六、推广意义",
                paper_data.get('impact', ''),
            ],
            font_prop,
        )

        if quality is not None and len(quality) > 0:
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            ax.axis("off")
            ax.set_title("数据质量与建模条件", fontproperties=font_prop, fontsize=18, color="#173b63", loc="left", pad=20)
            show = quality.copy()
            show.columns = ["站点", "阀门", "样本数", "起始日期", "结束日期", "缺失率(%)", "IF/LOF就绪", "突变检测就绪", "数据来源"]
            table = ax.table(cellText=show.astype(str).values, colLabels=show.columns, loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            for cell in table.get_celld().values():
                cell.set_text_props(fontproperties=font_prop)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()
