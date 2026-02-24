# 📈 Stock-Ward

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Update Log](https://img.shields.io/badge/Updates-View%20Log-orange)](./updates.md)

[English](../README.md) | [简体中文]

**优雅、高效的量化投资与企业估值工作站**

Stock-Ward 是一个基于 Python 和 Streamlit 精心打造的专业级财务分析平台。它致力于将繁杂的价值投资流程自动化——从全球市场数据抓取、财务报表标准化清洗，到严谨的估值模型（如 PE Band、DCF ）应用，一气呵成。

通过直观的数据可视化与理性的财务分析，帮助您拨开市场迷雾，发现真正被低估的优质资产。

---

## 📸 系统掠影

|                        仪表盘概览                        |                       深度估值模型                       |
| :------------------------------------------------------: | :------------------------------------------------------: |
| ![Dashboard](../assets/images/dashboard_placeholder.png) | ![Valuation](../assets/images/valuation_placeholder.png) |

> _让复杂沉闷的财务数据，转化为清晰可执行的投资洞察。_

---

## ✨ 核心亮点

### 🌍 全球视野

无缝管理与分析全球主流市场的上市公司，系统将自动处理不同货币单位（Billion/Million）与汇率转换。

- **US** (🇺🇸 美股)
- **CN** (🇨🇳 中国A股)
- **HK** (🇭🇰 港股)
- **JP** (🇯🇵 日本)
- **TW** (🇹🇼 台湾)

### 🧠 智能数据引擎

- **累计转单季 (SQ)**: 内置智能算法，自动将累计财务报表（如 Q3 YTD）精准拆解为单季度数据，让环比/同比趋势分析更加纯粹。
- **智能回填技术**: 深度集成 `yfinance`，根据财报实际发布日期，自动回溯并精准填充当时的市值与收盘价。

### 📊 完备的估值体系

- **PE Band 市盈率通道**: 独家计算历史 PE 分位点与标准差通道，一眼看穿当前价格的安全边际。
- **DCF 现金流折现模型**: 搭载自动化的 WACC 计算模块，支持 TTM 与 FY 双重预测基准，并可灵活调整增长率与终值假设进行敏感性分析。
- **高阶分析矩阵**: 提供 PEG 估值分析、 ROIC/ROE 质量因子拆解，以及分析师一致性预期追踪。

### 📈 沉浸式数据交互

- 采用 Plotly 驱动的动态图表，为您深度剖析营收、净利润、利润率与自由现金流的历史趋势，交互流畅顺滑。

---

## 🛠️ 稳健的技术架构

- **[Streamlit](https://streamlit.io/)**: 构筑极简、响应迅速且美观的 Web UI。
- **[Pandas](https://pandas.pydata.org/)**: 充当强大的金融数据清洗引擎。
- **[SQLite](https://www.sqlite.org/)**: 采用本地轻量级数据库，确保您的财务数据隐私绝对安全。
- **[yfinance](https://github.com/ranarousset/yfinance)**: 稳定接入实时与历史市场行情。

---

## 🚀 极速启航

只需简单几步，即可搭建您的专属基本面投研工作站：

### 环境要求

- Python 3.8 或更高版本
- Git 工具

### 安装指南

1. **克隆项目代码**

   ```bash
   git clone https://github.com/Seanyim/Stock-Ward.git
   cd Stock-Ward
   ```

2. **安装核心依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **启动应用程序**

   ```bash
   streamlit run main.py
   ```

4. **访问系统界面**
   在浏览器中打开网址：`http://localhost:8501` 即刻体验。

---

## 📖 工作流指南

我们推荐您按照以下标准工作流进行企业分析：

1. **添加目标公司** 🏢
   在侧边栏输入公司代码 (Ticker, 例如 `AAPL`, `600519.SS`, `9988.HK`)，选择对应的地区和单位，点击“添加/更新公司”。

2. **录入财务数据** 📝
   进入 **"数据录入"** 面板，输入或更新历史财报数据。系统将自动在后台为您补全缺失的市场价格数据。

3. **洞察业务趋势** 📉
   切换至 **"趋势分析"** 面板，观察营收增长率、毛利率、净利率等核心指标的长期走势，识别企业生命周期。

4. **执行绝对估值** 🧮
   使用 **"估值模型"** 面板，通过 DCF 或 PE 模型测算内在价值。您可以随时调整假设参数，进行各种悲观/乐观场景的压力测试。

---

## ⚙️ 个性化配置

- **网络代理**: 如果您所在地区访问 Yahoo Finance 受限，可在系统侧边栏一键配置 Proxy URL (默认: `http://127.0.0.1:10808`)。

---

## 📄 开源许可证

本项目基于 MIT 许可证开源 - 详细信息请参阅 [LICENSE](../LICENSE) 文件。
