import json
import os
import uuid
from typing import Iterable

import numpy as np
import pandas as pd

STATUS_FLOW = ["待确认", "已派工", "处理中", "已验证", "已关闭"]
STATIONS = ["华盘LNG加气站", "罗所LNG加气站"]
DEFAULT_STATION = "华盘LNG加气站"
DATA_SOURCE_OPTIONS = ["真实数据", "模拟数据", "未标注"]

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
    "data_source_tag",
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


def ensure_local_files(data_file: str, alert_file: str, audit_file: str) -> None:
    if not os.path.exists(data_file):
        pd.DataFrame(columns=BASE_DATA_COLS).to_csv(data_file, index=False, encoding="utf-8-sig")
    if not os.path.exists(alert_file):
        pd.DataFrame(columns=BASE_ALERT_COLS).to_csv(alert_file, index=False, encoding="utf-8-sig")
    if not os.path.exists(audit_file):
        pd.DataFrame(columns=BASE_AUDIT_COLS).to_csv(audit_file, index=False, encoding="utf-8-sig")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=BASE_DATA_COLS)

    rename_map = {
        "Valve_type": "valve_type",
        "P_now": "p_now",
        "P_max": "p_max",
        "Level": "level",
        "Temp": "temp",
        "PSV_act": "psv_act",
        "PSV_weeping": "psv_weeping",
    }
    df = df.rename(columns=rename_map).copy()

    for col in BASE_DATA_COLS:
        if col not in df.columns:
            if col == "station":
                df[col] = DEFAULT_STATION
            elif col == "data_source_tag":
                df[col] = "真实数据"
            elif col in ["operator_role", "operator_name", "updated_at"]:
                df[col] = ""
            else:
                df[col] = np.nan

    return df[BASE_DATA_COLS]


def normalize_data_df(df0: pd.DataFrame, stations: Iterable[str] | None = None, default_station: str = DEFAULT_STATION) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return pd.DataFrame(columns=BASE_DATA_COLS)

    stations = list(stations or STATIONS)
    df = standardize_columns(df0.copy())
    df["station"] = df["station"].fillna(default_station).replace("", default_station)
    df["station"] = df["station"].where(df["station"].isin(stations), default_station)
    df["data_source_tag"] = df["data_source_tag"].fillna("真实数据").replace("", "真实数据")
    df["data_source_tag"] = df["data_source_tag"].where(df["data_source_tag"].isin(DATA_SOURCE_OPTIONS), "未标注")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    for col in ["p_now", "p_max", "level", "temp", "psv_act", "psv_weeping"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["p_now"] = df["p_now"].clip(lower=0, upper=2)
    df["p_max"] = df["p_max"].clip(lower=0, upper=2)
    df["level"] = df["level"].clip(lower=0, upper=100)
    df["temp"] = df["temp"].clip(lower=-50, upper=80)

    bad_max = df["p_max"].notna() & df["p_now"].notna() & (df["p_max"] < df["p_now"])
    if bad_max.any():
        df.loc[bad_max, "p_max"] = df.loc[bad_max, "p_now"]

    df = df.dropna(subset=["date", "station", "valve_type", "p_now", "p_max"])
    df = (
        df.sort_values(["station", "valve_type", "date", "updated_at"])
        .drop_duplicates(subset=["date", "station", "valve_type"], keep="last")
        .reset_index(drop=True)
    )
    return df


def scope_filter(df: pd.DataFrame, station_scope: str) -> pd.DataFrame:
    if len(df) == 0 or station_scope == "ALL":
        return df.copy()
    return df[df["station"] == station_scope].copy()


def load_data(
    station_scope: str,
    role: str,
    *,
    use_supabase: bool,
    supabase,
    table_data: str,
    data_file: str,
) -> pd.DataFrame:
    if use_supabase and supabase is not None:
        raw = pd.DataFrame((supabase.table(table_data).select("*").execute().data or []))
    else:
        raw = pd.read_csv(data_file)
    return scope_filter(normalize_data_df(raw), station_scope)


def save_record(
    record: dict,
    station_scope: str,
    role: str,
    *,
    use_supabase: bool,
    supabase,
    table_data: str,
    data_file: str,
) -> None:
    if role == "leader":
        raise PermissionError("领导账号为只读，不允许写入数据")
    if station_scope != "ALL" and record.get("station") != station_scope:
        raise PermissionError("只能写入本站数据")

    payload = record.copy()
    payload["updated_at"] = pd.Timestamp.now().isoformat()
    payload["data_source_tag"] = payload.get("data_source_tag") or "真实数据"

    if use_supabase and supabase is not None:
        supabase.table(table_data).upsert(payload, on_conflict="date,station,valve_type").execute()
        return

    current = normalize_data_df(pd.read_csv(data_file))
    merged = normalize_data_df(pd.concat([current, pd.DataFrame([payload])], ignore_index=True))
    merged.to_csv(data_file, index=False, encoding="utf-8-sig")


def normalize_alert_df(df0: pd.DataFrame) -> pd.DataFrame:
    if df0 is None or len(df0) == 0:
        return pd.DataFrame(columns=BASE_ALERT_COLS)

    df = df0.copy()
    for col in BASE_ALERT_COLS:
        if col not in df.columns:
            df[col] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["station"] = df["station"].fillna(DEFAULT_STATION).replace("", DEFAULT_STATION)
    df["status"] = df["status"].replace("", "待确认")
    return df[BASE_ALERT_COLS]


def _safe_num(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def normalize_trigger_detail(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, (int, float, np.number)):
                out[key] = _safe_num(item)
            else:
                out[key] = item
        return out
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception:
            return {"raw": stripped}
    return {}


def append_audit(
    entity_type: str,
    entity_id: str,
    action: str,
    operator: str,
    payload: str,
    *,
    use_supabase: bool,
    supabase,
    table_audit: str,
    audit_file: str,
) -> None:
    log = {
        "id": str(uuid.uuid4()),
        "entity_type": entity_type,
        "entity_id": entity_id,
        "action": action,
        "operator": operator,
        "payload": payload,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    if use_supabase and supabase is not None:
        supabase.table(table_audit).insert(log).execute()
    else:
        raw = pd.read_csv(audit_file)
        pd.concat([raw, pd.DataFrame([log])], ignore_index=True).to_csv(audit_file, index=False, encoding="utf-8-sig")


def _find_alert(alerts: pd.DataFrame, date_value, station: str, valve_type: str) -> pd.DataFrame:
    return alerts[
        (alerts["date"] == pd.to_datetime(date_value).date())
        & (alerts["station"] == station)
        & (alerts["valve_type"] == valve_type)
    ]


def _load_alerts_all(*, use_supabase: bool, supabase, table_alert: str, alert_file: str) -> pd.DataFrame:
    if use_supabase and supabase is not None:
        raw = pd.DataFrame((supabase.table(table_alert).select("*").execute().data or []))
    else:
        raw = pd.read_csv(alert_file)
    return normalize_alert_df(raw)


def create_or_update_alert(
    record: dict,
    *,
    use_supabase: bool,
    supabase,
    table_alert: str,
    alert_file: str,
) -> None:
    now_iso = pd.Timestamp.now().isoformat()
    alerts = _load_alerts_all(use_supabase=use_supabase, supabase=supabase, table_alert=table_alert, alert_file=alert_file)
    found = _find_alert(alerts, record["date"], record["station"], record["valve_type"])

    if len(found) > 0:
        idx = found.index[0]
        alerts.loc[idx, "risk_level"] = record.get("risk_level", alerts.loc[idx, "risk_level"])
        alerts.loc[idx, "trigger_source"] = record.get("trigger_source", alerts.loc[idx, "trigger_source"])
        alerts.at[idx, "trigger_detail"] = json.dumps(
            normalize_trigger_detail(record.get("trigger_detail", alerts.loc[idx, "trigger_detail"])),
            ensure_ascii=False,
        )
        alerts.loc[idx, "updated_at"] = now_iso
        alerts.loc[idx, "status"] = alerts.loc[idx, "status"] or "待确认"
    else:
        new_alert = {
            "id": str(uuid.uuid4()),
            "date": str(pd.to_datetime(record["date"]).date()),
            "station": record["station"],
            "valve_type": record["valve_type"],
            "risk_level": record.get("risk_level", "🔴 高风险"),
            "trigger_source": record.get("trigger_source", "rule"),
            "trigger_detail": json.dumps(normalize_trigger_detail(record.get("trigger_detail", {})), ensure_ascii=False),
            "status": "待确认",
            "owner": "",
            "action_taken": "",
            "verification_result": "",
            "created_at": now_iso,
            "updated_at": now_iso,
            "closed_at": "",
        }
        alerts = pd.concat([alerts, pd.DataFrame([new_alert])], ignore_index=True)

    if use_supabase and supabase is not None:
        record_date = str(pd.to_datetime(record["date"]).date())
        base_payload = {
            "date": record_date,
            "station": record["station"],
            "valve_type": record["valve_type"],
            "risk_level": record.get("risk_level", "🔴 高风险"),
            "trigger_source": record.get("trigger_source", "rule"),
            "trigger_detail": normalize_trigger_detail(record.get("trigger_detail", {})),
            "updated_at": now_iso,
        }
        query = (
            supabase.table(table_alert)
            .select("id,status,owner,action_taken,verification_result,created_at,closed_at")
            .eq("date", record_date)
            .eq("station", record["station"])
            .eq("valve_type", record["valve_type"])
            .limit(1)
            .execute()
        )
        current = query.data or []
        if current:
            old = current[0]
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
        supabase.table(table_alert).upsert(payload, on_conflict="date,station,valve_type").execute()
    else:
        alerts.to_csv(alert_file, index=False, encoding="utf-8-sig")


def update_alert_status(
    alert_id: str,
    new_status: str,
    operator: str,
    action_taken: str = "",
    verification_result: str = "",
    *,
    use_supabase: bool,
    supabase,
    table_alert: str,
    table_audit: str,
    alert_file: str,
    audit_file: str,
) -> None:
    alerts = _load_alerts_all(use_supabase=use_supabase, supabase=supabase, table_alert=table_alert, alert_file=alert_file)
    hit = alerts[alerts["id"].astype(str) == str(alert_id)]
    if len(hit) == 0:
        raise ValueError("未找到告警")

    idx = hit.index[0]
    current_status = alerts.loc[idx, "status"] if alerts.loc[idx, "status"] in STATUS_FLOW else "待确认"
    if new_status not in STATUS_FLOW:
        raise ValueError("非法状态")
    if STATUS_FLOW.index(new_status) not in [STATUS_FLOW.index(current_status), STATUS_FLOW.index(current_status) + 1]:
        raise ValueError("状态仅允许保持不变或推进一步")
    if new_status == "已关闭" and (not action_taken.strip() or not verification_result.strip()):
        raise ValueError("关闭告警前必须填写整改措施和复验结果")

    if action_taken.strip():
        alerts.loc[idx, "action_taken"] = action_taken.strip()
    if verification_result.strip():
        alerts.loc[idx, "verification_result"] = verification_result.strip()
    alerts.loc[idx, "status"] = new_status
    alerts.loc[idx, "owner"] = operator
    alerts.loc[idx, "updated_at"] = pd.Timestamp.now().isoformat()
    if new_status == "已关闭":
        alerts.loc[idx, "closed_at"] = pd.Timestamp.now().isoformat()

    if use_supabase and supabase is not None:
        updates = {
            "status": new_status,
            "owner": operator,
            "updated_at": alerts.loc[idx, "updated_at"],
            "action_taken": alerts.loc[idx, "action_taken"],
            "verification_result": alerts.loc[idx, "verification_result"],
        }
        if new_status == "已关闭":
            updates["closed_at"] = alerts.loc[idx, "closed_at"]
        supabase.table(table_alert).update(updates).eq("id", str(alert_id)).execute()
    else:
        alerts.to_csv(alert_file, index=False, encoding="utf-8-sig")

    append_audit(
        entity_type="alert",
        entity_id=str(alert_id),
        action=f"status:{current_status}->{new_status}",
        operator=operator,
        payload=f"action_taken={action_taken}; verification_result={verification_result}",
        use_supabase=use_supabase,
        supabase=supabase,
        table_audit=table_audit,
        audit_file=audit_file,
    )


def list_alerts(
    station_scope: str,
    role: str,
    *,
    use_supabase: bool,
    supabase,
    table_alert: str,
    alert_file: str,
) -> pd.DataFrame:
    alerts = _load_alerts_all(use_supabase=use_supabase, supabase=supabase, table_alert=table_alert, alert_file=alert_file)
    return scope_filter(alerts, station_scope).sort_values(["status", "date"], ascending=[True, False]).reset_index(drop=True)
