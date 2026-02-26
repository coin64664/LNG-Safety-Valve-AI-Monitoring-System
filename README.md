# LNG-Safety-Valve-AI-Monitoring-System
LNG Safety Valve AI Monitoring System v0.2
🚀 LNG Safety Valve Intelligent Risk Prediction System
Overview

This project is an AI-driven monitoring and predictive risk assessment system for LNG (Liquefied Natural Gas) station safety valves.

The system combines:

Engineering-based health index modeling

Unsupervised anomaly detection (Isolation Forest)

Trend analysis

Risk scoring

Visual decision dashboards

Its goal is to transform daily manual operational records into an intelligent decision-support tool for preventive maintenance and operational safety management.

🎯 Why This Project Matters

In LNG stations, safety valves play a critical role in:

Preventing overpressure events

Protecting storage tanks and pipelines

Avoiding costly emergency shutdowns

Traditional monitoring methods rely on:

Static threshold checks

Manual inspection

Reactive maintenance

This system shifts from reactive monitoring to predictive risk management.

🧠 Core Features
1️⃣ Health Index (HI) Modeling

A dynamic health score (0–100) is calculated daily for each valve based on:

Pressure ratio vs set pressure

3-day pressure trend slope

Valve activation events

Micro-leak events

High temperature & high liquid level coupling effects

The system converts raw operational data into a clear, decision-oriented risk level:

🟢 Safe

🟡 Warning

🔴 High Risk

2️⃣ AI-Based Anomaly Detection

An Isolation Forest model is applied per valve to detect abnormal operational patterns without requiring labeled failure data.

Features used:

Current pressure

Daily max pressure

Liquid level

Ambient temperature

Pressure ratio

Trend slope

Valve activity count

AI anomaly signals are fused into the final health index, enhancing detection beyond rule-based logic.

3️⃣ Intelligent Visualization Dashboard

The system provides:

Single-valve pressure & health trends

Health heatmap across valves and time

Valve comparison analytics

Pressure vs activity correlation analysis

Executive-friendly decision view

The dashboard is optimized for management-level interpretation.

4️⃣ Data Integrity & Validation

To ensure reliability:

Automatic correction of physical inconsistencies (p_max ≥ p_now)

Duplicate daily records auto-merged

Value range validation

Optional Supabase cloud database integration

🔮 Future Extensions (Roadmap)

This architecture supports expansion toward:

Remaining useful life (RUL) estimation

7-day risk probability forecasting

Maintenance prioritization ranking

SHAP-based AI explainability

Economic risk modeling

Automated PDF reporting

🏗️ System Architecture

Frontend:

Streamlit interactive dashboard

Backend:

Pandas data processing

Scikit-learn (Isolation Forest)

Matplotlib visualization

Storage:

Local CSV (development)

Supabase PostgreSQL (cloud deployment)

📊 Target Users

LNG station operators

Safety engineers

Maintenance planners

Operational managers

💡 Key Concept

This system is not just a monitoring tool.

It is designed as a Predictive Maintenance & Risk Decision Support Platform.

It answers three core management questions:

Is there a risk developing?

How likely is a failure in the near future?

Which valve should be prioritized for inspection?

📦 Deployment

Run locally:

streamlit run psv_app.py


Optional cloud database setup:

Configure SUPABASE_URL

Configure SUPABASE_KEY

Enable cloud storage mode

📌 Technical Highlights

Hybrid rule-based + AI anomaly detection model

Time-series slope-based degradation detection

Lightweight industrial AI architecture

Designed for explainability and practical adoption
