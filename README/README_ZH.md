# PriceEstimated 📈 

[English](../README.md) | [简体中文] | 

**高效率量化投资估值工作站**

PriceEstimated 是一个基于 Python 的专业级财务分析工具。它实现了市场数据抓取、财务报表标准化以及严谨估值模型（PE Band 与 DCF）的自动化流转，旨在通过数据驱动发现投资价值。

---

## 🎯 核心功能

* **🧩 累计转单季 (SQ)**: 内置复杂逻辑，自动将财报中的累计值（如 H1, Q1-Q3）转换为独立的单季度数据，确保趋势分析的准确性。
* **📊 双重估值体系**:
    * **PE Band (市盈率带)**: 基于历史 PE 分位点与前瞻 PE (Forward PE) 进行统计学估值。
    * **DCF (现金流折现)**: 通过自动化 WACC (加权平均资本成本) 计算推导内在价值。
* **⚡ 智能数据补全**: 调用 `yfinance` 自动回填历史财报披露日的市值与收盘价，告别繁琐的手工录入。
* **📈 交互式可视化**: 使用 Plotly 构建动态 PE Band 曲线和财务趋势图表。
* **🗄️ 本地数据库**: 基于 SQLite 的高性能后端，持久化存储标准化后的财务记录。

## 🛠️ 技术栈

* **界面框架**: [Streamlit](https://streamlit.io/)
* **数据引擎**: [Pandas](https://pandas.pydata.org/), [SQLite](https://www.sqlite.org/)
* **金融接口**: [yfinance](https://github.com/ranarousset/yfinance)
* **可视化**: [Plotly](https://plotly.com/python/)

## 📊 量化逻辑

核心估值基于内在价值 ($V$) 公式：

$$V = \sum_{t=1}^{n} \frac{CF_t}{(1 + r)^t} + \frac{TV}{(1 + r)^n}$$

其中 $CF_t$ 为自由现金流，$r$ 为计算得出的 WACC，$TV$ 为终值。

## 🚀 快速开始

1. **克隆与安装**:
   ```bash
   git clone [https://github.com/Seanyim/PriceEstimated.git](https://github.com/Seanyim/PriceEstimated.git)
   cd PriceEstimated
   pip install -r requirements.txt
   ```
