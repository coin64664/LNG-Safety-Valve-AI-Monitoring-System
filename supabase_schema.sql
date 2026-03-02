-- Supabase schema for multi-site + role-based LNG PSV monitoring (phase 1)

-- 1) psv_data: add station/operation metadata
alter table if exists public.psv_data
  add column if not exists station text,
  add column if not exists operator_role text,
  add column if not exists operator_name text,
  add column if not exists updated_at timestamptz;

-- Existing history defaults to Huapan site
update public.psv_data
set station = '华盘LNG加气站'
where station is null or station = '';

-- Site whitelist and unique key
alter table if exists public.psv_data
  add constraint if not exists psv_data_station_check
  check (station in ('华盘LNG加气站', '罗所LNG加气站'));

create unique index if not exists psv_data_uniq_date_station_valve
  on public.psv_data(date, station, valve_type);

-- 2) psv_alerts: alert lifecycle
create table if not exists public.psv_alerts (
  id uuid primary key default gen_random_uuid(),
  date date not null,
  station text not null,
  valve_type text not null,
  risk_level text,
  trigger_source text,
  trigger_detail jsonb,
  status text not null default '待确认',
  owner text,
  action_taken text,
  verification_result text,
  created_at timestamptz not null default now(),
  updated_at timestamptz,
  closed_at timestamptz,
  constraint psv_alerts_station_check check (station in ('华盘LNG加气站', '罗所LNG加气站'))
);

create unique index if not exists psv_alerts_uniq_date_station_valve
  on public.psv_alerts(date, station, valve_type);

-- 3) psv_audit_logs: immutable operation trail
create table if not exists public.psv_audit_logs (
  id uuid primary key default gen_random_uuid(),
  entity_type text not null,
  entity_id text not null,
  action text not null,
  operator text,
  payload jsonb,
  created_at timestamptz not null default now()
);
