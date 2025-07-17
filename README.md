# 🪙 Crypto Price Analysis Dashboard

An end-to-end project that covers the **full data analytics pipeline** — from data wrangling and machine learning to database integration and Power BI visualization — focused on analyzing and predicting cryptocurrency price trends.

## 📌 Project Overview

This project aims to build a **Crypto Price Dashboard** by combining:
- Merging raw CSV datasets of 20+ coins
- Applying **machine learning models** (KNN, Decision Tree)
- Storing & querying data in **MySQL**
- Visualizing trends and predictions in **Power BI**

---

## ⚙️ Technologies Used

| Tool / Language | Purpose |
|----------------|---------|
| **Python** | Data preprocessing, merging, modeling |
| **pandas, seaborn, matplotlib** | Data cleaning, feature engineering, plotting |
| **scikit-learn** | Machine Learning (KNN, Decision Tree) |
| **MySQL** | Data import, queries, and relational storage |
| **Power BI** | Interactive dashboard and visual storytelling |

---

## 📁 Workflow Summary

### 🧩 1. Data Preparation
- Merged 20+ CSVs (each representing a crypto coin) into one consolidated dataset
- Cleaned columns: `Open`, `Close`, `High`, `Low`, `Volume`, `Marketcap`, `%_Change`, `Volatility`, etc.
- Handled missing values and standardized formats

### 🤖 2. Machine Learning
- Trained **K-Nearest Neighbors** and **Decision Tree** models
- Predicted short-term **% change** and **price volatility**
- Visualized prediction results using **matplotlib** and **seaborn**

### 🗃️ 3. Database Integration
- Imported merged dataset into **MySQL**
- Designed schema with appropriate datatypes and indices
- Performed SQL queries to prepare filtered views for Power BI

### 📊 4. Power BI Dashboard
- Connected to MySQL database
- Created interactive visuals:
  - **Cards** for total coins, avg % change, highest volatility
  - **Slicers** for coin, date range, volatility filter
  - **Line/bar/area charts** for trend analysis
- Applied:
  - Custom DAX measures
  - Borders, themes, and icon effects
  - Interactive filters and dynamic KPI views

---

## 📈 ML Model Accuracy

| Model | Target | Accuracy / Result |
|-------|--------|-------------------|
| KNN | %_Change | ~52%              |
| Decision Tree | Volatility | ~53%  |

---

## 🔗 How to Run

### 🐍 Python
```bash
pip install pandas matplotlib seaborn scikit-learn mysql-connector-python
python python.py
