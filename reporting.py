from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


PATENT_HIGHLIGHTS = [
    "面向LNG安全阀的机理健康指数构建方法",
    "基于站点-阀门分组的滚动自适应历史基线",
    "结合孤立森林与时序退化特征的双引擎异常识别",
    "基于持续性与异常共识的分层风险升级方法",
    "基于异常原因链的闭环处置建议生成方法",
]


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["station", "valve_type", "samples", "date_start", "date_end", "missing_rate", "ai_ready", "data_source_tag"])

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
            ai_ready=("date", lambda s: len(s) >= 30),
            data_source_tag=("data_source_tag", lambda s: " / ".join(sorted(pd.Series(s).dropna().astype(str).unique()))),
        )
        .reset_index()
    )
    quality["missing_rate"] = (quality["missing_rate"] * 100).round(1)
    return quality


def build_patent_overview() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "创新模块": [f"创新点{i}" for i in range(1, len(PATENT_HIGHLIGHTS) + 1)],
            "方法描述": PATENT_HIGHLIGHTS,
        }
    )


def _top_reason_text(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or "ai_reason_top1" not in df.columns:
        return "暂无"
    values = (
        df.loc[df["ai_reason_top1"].astype(str) != "-", "ai_reason_top1"]
        .value_counts()
        .head(3)
        .to_dict()
    )
    return "、".join([f"{k}:{v}次" for k, v in values.items()]) if values else "暂无"


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
            "项目名称：一种基于自适应健康指数与分层异常共识的LNG储罐安全阀智能预警方法",
            f"报告范围：{date_start} 至 {date_end}",
            f"统计范围：{station_pick}",
            f"平均HI_final：{avg_hi:.1f}",
            f"高风险记录数：{red_cnt}",
            f"预警记录数：{warn_cnt}",
            f"AI观察记录数：{observe_cnt}",
            f"AI升级/高风险记录数：{upgrade_cnt}",
            f"主要异常原因：{_top_reason_text(df)}",
            f"告警闭环率：{close_rate:.1f}%",
            storyline,
            "管理建议：优先复盘连续异常且接近整定压力的阀门，结合闭环处置结果安排维护验证。",
        ]
    )


def generate_competition_brief(df: pd.DataFrame, validation_summary: dict, case_replay: dict) -> str:
    if df is None or len(df) == 0:
        return "当前暂无足够数据生成比赛摘要。"

    focus_valve = case_replay.get("valve_type", "暂无")
    focus_reason = case_replay.get("focus_reason", "暂无")
    return "\n".join(
        [
            "项目亮点摘要",
            "1. 面向LNG安全阀场景，构建规则机理 + 自适应基线 + 时序退化 + 异常共识的复合智能预警方法。",
            f"2. 当前纳入评估阀门组 {validation_summary.get('group_count', 0)} 个，其中满足AI建模条件 {validation_summary.get('eligible_group_count', 0)} 个。",
            f"3. 已识别AI观察异常 {validation_summary.get('observe_count', 0)} 条，升级/高风险 {validation_summary.get('escalate_count', 0) + validation_summary.get('high_count', 0)} 条。",
            f"4. 代表性案例阀门：{focus_valve}，系统结论：{focus_reason}",
            "5. 系统支持站点隔离、告警闭环、管理摘要导出，可直接服务比赛答辩和后续推广。",
        ]
    )


def generate_patent_disclosure_outline(df: pd.DataFrame, cases: Iterable[dict]) -> str:
    case_list = [case for case in cases if case]
    case_desc = "；".join(
        [
            f"{case.get('station', '-')}-{case.get('valve_type', '-')}: {case.get('focus_date', '-')} 触发 {case.get('focus_stage', '-')}"
            for case in case_list
        ]
    ) or "待补充实施例"

    return "\n".join(
        [
            "# 技术交底书提纲",
            "一、背景技术",
            "现有LNG安全阀巡检多依赖静态阈值和人工经验，难以识别连续退化和隐蔽异常。",
            "二、现有不足",
            "传统方法难以同时表达机理风险、历史基线偏离、异常持续性以及闭环处置价值。",
            "三、发明目的",
            "提出一种基于自适应健康指数与分层异常共识的LNG储罐安全阀智能预警方法，实现更早识别、更可解释、更易闭环。",
            "四、方法流程",
            "1. 获取安全阀日常运行数据；",
            "2. 构建机理健康指数；",
            "3. 建立站点-阀门分组的滚动历史基线；",
            "4. 提取时序退化特征并结合孤立森林识别异常；",
            "5. 通过持续性与异常共识机制输出分层风险结果；",
            "6. 根据异常原因链生成处置建议并形成闭环记录。",
            "五、核心创新点",
            *[f"{idx + 1}. {item}" for idx, item in enumerate(PATENT_HIGHLIGHTS)],
            "六、具体实施例",
            case_desc,
            "七、可替代实施方式",
            "孤立森林可替换为其他无监督异常检测器，但保留机理健康指数、历史基线、时序退化与分层共识决策主链。",
        ]
    )
