from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ai_engine import build_case_replay, run_scoring_pipeline
from data_pipeline import normalize_alert_df, normalize_data_df
from reporting import build_data_quality_table, generate_technical_paper_data, render_technical_paper_pdf

try:
    from supabase import create_client

    SUPABASE_OK = True
except Exception:
    SUPABASE_OK = False


DEFAULT_DATA_FILE = "psv_data.csv"
DEFAULT_ALERT_FILE = "psv_alerts.csv"
DEFAULT_OUTPUT = "outputs/lng_psv_technical_paper.pdf"
TABLE_DATA = "psv_data"
TABLE_ALERT = "psv_alerts"


def _load_local_csv(data_file: str, alert_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_df = normalize_data_df(pd.read_csv(data_file)) if Path(data_file).exists() else pd.DataFrame()
    alert_df = normalize_alert_df(pd.read_csv(alert_file)) if Path(alert_file).exists() else pd.DataFrame()
    return data_df, alert_df


def _load_supabase() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SUPABASE_OK:
        raise RuntimeError("当前环境未安装 supabase 依赖，无法从 Supabase 读取数据。")

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError("未检测到 SUPABASE_URL / SUPABASE_KEY。")

    client = create_client(url, key)
    data_df = normalize_data_df(pd.DataFrame((client.table(TABLE_DATA).select("*").execute().data or [])))
    alert_df = normalize_alert_df(pd.DataFrame((client.table(TABLE_ALERT).select("*").execute().data or [])))
    return data_df, alert_df


def _pick_focus_case(df: pd.DataFrame, station: str | None, valve_type: str | None) -> tuple[str, str]:
    work = df.copy()
    if station:
        work = work[work["station"] == station].copy()
    if valve_type:
        work = work[work["valve_type"] == valve_type].copy()

    if len(work) == 0:
        raise RuntimeError("指定的站点/阀门范围内没有可用记录，无法生成技术论文。")

    focus = work.sort_values(["consensus_score", "date"], ascending=[False, False]).iloc[0]
    return str(focus["station"]), str(focus["valve_type"])


def main():
    parser = argparse.ArgumentParser(description="生成 LNG 安全阀系统技术论文 PDF。")
    parser.add_argument("--source", choices=["local", "supabase"], default="local", help="数据来源，默认 local。")
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE, help="本地监测数据 CSV 路径。")
    parser.add_argument("--alert-file", default=DEFAULT_ALERT_FILE, help="本地告警数据 CSV 路径。")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 PDF 路径。")
    parser.add_argument("--station", default="", help="可选，指定站点。")
    parser.add_argument("--valve", default="", help="可选，指定阀门。")
    parser.add_argument("--start-date", default="", help="可选，开始日期，格式 YYYY-MM-DD。")
    parser.add_argument("--end-date", default="", help="可选，结束日期，格式 YYYY-MM-DD。")
    parser.add_argument("--contamination", type=float, default=0.08, help="多模型异常灵敏度，默认 0.08。")
    parser.add_argument("--disable-ai", action="store_true", help="禁用高阶 AI，仅用规则与退化趋势生成材料。")
    args = parser.parse_args()

    if args.source == "supabase":
        data_df, alert_df = _load_supabase()
    else:
        data_df, alert_df = _load_local_csv(args.data_file, args.alert_file)

    if data_df is None or len(data_df) == 0:
        raise RuntimeError("没有读到监测数据，无法生成技术论文。")

    scored_df = run_scoring_pipeline(data_df, enable_ai=not args.disable_ai, contamination=float(args.contamination))

    if args.start_date:
        start_date = pd.to_datetime(args.start_date, errors="raise").date()
        scored_df = scored_df[scored_df["date"] >= start_date].copy()
        if len(alert_df) > 0:
            alert_df = alert_df[pd.to_datetime(alert_df["date"], errors="coerce").dt.date >= start_date].copy()
    if args.end_date:
        end_date = pd.to_datetime(args.end_date, errors="raise").date()
        scored_df = scored_df[scored_df["date"] <= end_date].copy()
        if len(alert_df) > 0:
            alert_df = alert_df[pd.to_datetime(alert_df["date"], errors="coerce").dt.date <= end_date].copy()

    if args.station:
        scored_df = scored_df[scored_df["station"] == args.station].copy()
        if len(alert_df) > 0:
            alert_df = alert_df[alert_df["station"] == args.station].copy()

    if len(scored_df) == 0:
        raise RuntimeError("筛选后没有可用数据，无法生成技术论文。")

    focus_station, focus_valve = _pick_focus_case(scored_df, args.station or None, args.valve or None)
    case_replay = build_case_replay(scored_df, focus_station, focus_valve)
    quality_table = build_data_quality_table(scored_df)
    paper_data = generate_technical_paper_data(scored_df, alert_df, case_replay, quality_table)
    pdf_bytes = render_technical_paper_pdf(paper_data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)

    print(f"技术论文 PDF 已生成：{out_path.resolve()}")
    print(f"案例站点：{focus_station}")
    print(f"案例阀门：{focus_valve}")
    print(f"数据范围：{scored_df['date'].min()} 至 {scored_df['date'].max()}")


if __name__ == "__main__":
    main()
