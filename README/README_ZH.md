# Stock-Ward 📈

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Update Log](https://img.shields.io/badge/Updates-View%20Log-orange)](./updates.md)

[English](../README.md) | [简体中文] |

**高效率量化投资与企业估值工作站**

Stock-Ward 是一个基于 Python 和 Streamlit 构建的专业级财务分析工具。它实现了从全球市场数据抓取、财务报表标准化清洗，到应用严谨的估值模型（PE Band, DCF 等）的全流程自动化，帮助投资者通过数据驱动发现被低估的优质资产。

---

## 📸 系统截图

|                          仪表盘                          |                         估值模型                         |
| :------------------------------------------------------: | :------------------------------------------------------: |
| ![Dashboard](../assets/images/dashboard_placeholder.png) | ![Valuation](../assets/images/valuation_placeholder.png) |

> _可视化财务趋势与内在价值分析_

---

## 🎯 核心功能

### 1. 全球市场支持 🌍

- 无缝管理主流市场的上市公司：
  - **US** (🇺🇸 美国)
  - **CN** (🇨🇳 中国 A 股)
  - **HK** (🇭🇰 香港)
  - **JP** (🇯🇵 日本)
  - **TW** (🇹🇼 台湾)
- 自动处理货币单位（Billion/Million）与汇率转换。

### 2. 智能数据管理 🧠

- **累计转单季 (SQ)**: 内置复杂算法，自动解构累计财务报表（如 Q3 YTD）为独立的单季度数据，确保季度环比/同比分析的准确性。
- **智能回填**: 集成 `yfinance` 接口，根据财报发布日期自动回溯并填充当时的市值与收盘价。

### 3. 完整的估值体系 📊

- **PE Band 市盈率带**:
  - 统计历史 PE 分位点与标准差通道。
  - 直观判断当前价格的安全边际。
- **DCF 现金流折现**:
  - 自动化的 WACC (加权平均资本成本) 计算模块。
  - 支持 TTM (滚动十二个月) 与 FY (会计年度) 两种预测基准。
  - 灵活调整增长率与终值假设。
- **高级模型**:
  - PEG 估值分析。
  - ROIC / ROE 质量因子分析。
- **分析师预期**: 跟踪市场一致性预期。

### 4. 交互式可视化 📈

- 基于 Plotly 的动态图表。
- 深度分析营收、净利润、利润率与自由现金流的历史趋势。

---

## 🛠️ 技术架构

- **前端**: [Streamlit](https://streamlit.io/) - 极简、高效的响应式 Web UI。
- **数据处理**: [Pandas](https://pandas.pydata.org/) - 核心金融数据清洗引擎。
- **存储**: [SQLite](https://www.sqlite.org/) - 本地轻量级数据库，数据隐私安全。
- **数据源**: [yfinance](https://github.com/ranarousset/yfinance) - 实时行情接入。

---

## 🚀 快速开始

### 环境要求

- Python 3.8 或更高版本
- Git

### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/Seanyim/Stock-Ward.git
   cd Stock-Ward
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **启动应用**

   ```bash
   streamlit run main.py
   ```

4. **访问系统**
   在浏览器中打开 `http://localhost:8501`。

---

## 📖 使用指南

### 1. 添加公司

- 在 **侧边栏** 输入代码 (Ticker) 与名称。
- 选择所属 **地区 (Region)** 与 **单位 (Unit)**。
- 点击 **"添加/更新公司"**。

### 2. 录入数据

- 进入 **"📝 数据录入"** 标签页。
- 输入或更新历史财务数据。
- 系统将自动尝试补全缺失的市场价格数据。

### 3. 趋势分析

- 切换至 **"📈 趋势分析"**。
- 观察营收增长率、毛利率、净利率等核心指标的长期走势。

### 4. 执行估值

- 使用 **"🧮 估值模型"** 测算合理价值。
- 在 DCF 模型中调整 WACC 参数与增长假设，进行敏感性分析。

---

## ⚙️ 配置说明

- **代理设置**: 如需通过代理访问 Yahoo Finance，可在侧边栏配置 Proxy URL (默认: `http://127.0.0.1:10808`)。

## 📄 许可证

本项目采用 MIT 许可证 - 详情见 [LICENSE](../LICENSE) 文件。
