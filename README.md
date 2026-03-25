# LNG Safety Valve AI Monitoring System

## 项目定位
本项目是一个面向 LNG 储罐安全阀场景的工业智能监测与风险预警系统，重点解决日常运行记录分散、阀门退化不易提前识别、告警缺少闭环跟踪的问题。

系统定位是“可直接使用的成熟产品”，主界面面向站点操作人员和管理者，核心关注运行状态、风险趋势、异常识别和处置建议。

## 当前能力
- 双站点分角色登录：华盘站、罗所站、领导
- 站点级数据隔离与领导只读查看
- 历史分析首屏展示：3线压力趋势图、HI 热力图、阀门 HI 对比
- 多模型异常识别：Isolation Forest、局部离群因子、时序突变检测、退化趋势引擎
- 统一风险分层：AI观察、AI升级、AI高风险
- 重点阀门健康档案与自动处置建议
- 告警闭环流转：待确认 -> 已派工 -> 处理中 -> 已验证 -> 已关闭
- 管理摘要、监测明细、告警明细、AI异常明细导出
- 本地 CSV 与 Supabase 双存储兼容

## AI 方法
系统当前采用“规则机理 + 自适应基线 + 时序退化 + 风险共识”的多模型识别链路：

1. 机理健康层  
基于压力接近整定值、近3日斜率、动作、微放散、温度液位耦合，形成基础 HI 与规则风险。

2. 自适应基线层  
按 `station + valve_type` 建立滚动历史基线，衡量“当前状态相对本站本阀历史是否异常”。

3. 时序退化层  
识别连续上升、持续接近整定值、短时突增、波动放大和活动异常等退化迹象。

4. 风险共识层  
融合 Isolation Forest、局部离群因子、时序突变检测、退化趋势引擎和规则风险，输出统一风险阶段与原因链。

## 项目结构
- [psv_app.py](./psv_app.py)：Streamlit 应用入口与页面编排
- [data_pipeline.py](./data_pipeline.py)：数据标准化、站点隔离、告警与审计
- [risk_engine.py](./risk_engine.py)：机理健康分与领域特征
- [ai_engine.py](./ai_engine.py)：多模型异常识别、风险共识、案例回放
- [reporting.py](./reporting.py)：管理摘要、数据质量和外部技术材料生成接口
- `tests/`：最小测试集

## 运行方式
本地运行：

```bash
streamlit run psv_app.py
```

如需启用云端数据：
- 配置 `SUPABASE_URL`
- 配置 `SUPABASE_KEY`
- 在 Supabase 中执行 [supabase_schema.sql](./supabase_schema.sql)

## 技术论文 PDF 导出
系统主界面不直接展示申报类内容；如需生成技术论文 PDF，可在项目根目录执行：

```bash
py -3 generate_technical_paper.py
```

默认会读取本地 `psv_data.csv` 与 `psv_alerts.csv`，输出到：

`outputs/lng_psv_technical_paper.pdf`

也可以指定范围，例如：

```bash
py -3 generate_technical_paper.py --station "华盘LNG加气站" --valve "储罐主阀" --start-date 2026-02-24 --end-date 2026-03-25
```

## 测试
```bash
py -3 -m unittest discover -s tests -v
```

## 数据说明
- 当前真实可用数据以华盘站为主
- 罗所站主要用于多站点架构展示与后续扩展
- 系统保留 `data_source_tag` 字段，用于区分真实数据、模拟数据和未标注数据

## 外部材料
如需继续扩展答辩材料、技术交底书或投稿材料，可以在 `reporting.py` 与 `generate_technical_paper.py` 基础上继续完善。
