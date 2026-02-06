# Stock-Ward 📈

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Update Log](https://img.shields.io/badge/Updates-View%20Log-orange)](./README/updates.md)

[English] | [简体中文](./README/README_ZH.md) |

**A High-Performance Quantitative Investment & Valuation Station**

Stock-Ward is a professional-grade financial analysis tool built with Python and Streamlit. It automates the entire workflow of value investing: from fetching global market data and normalizing financial statements to applying rigorous valuation models (PE Bands, DCF, etc.) to identify undervalued assets.

---

## 📸 Screenshots

|                       Dashboard                       |                    Valuation Model                    |
| :---------------------------------------------------: | :---------------------------------------------------: |
| ![Dashboard](assets/images/dashboard_placeholder.png) | ![Valuation](assets/images/valuation_placeholder.png) |

> _Stock-Ward providing visualized financial trends and intrinsic value analysis._

---

## 🎯 Key Features

### 1. Multi-Market Support 🌍

- Seamlessly manage companies across major global markets:
  - **US** (🇺🇸 United States)
  - **CN** (🇨🇳 China A-Shares)
  - **HK** (🇭🇰 Hong Kong)
  - **JP** (🇯🇵 Japan)
  - **TW** (🇹🇼 Taiwan)
- Automatic currency handling and unit conversion (Billion/Million).

### 2. Intelligent Data Management �

- **Cumulative to Single Quarter (SQ)**: Built-in algorithms automatically decrypt cumulative financial reports (e.g., Q3 YTD) into discrete single-quarter data for precise trend analysis.
- **Smart Backfilling**: Integrates with `yfinance` to automatically fetch and fill historical market caps and stock prices based on report dates.

### 3. Comprehensive Valuation System 📊

- **PE Band Analysis**:
  - Visualizes historical PE ratios with standard deviation bands.
  - Calculates percentile rankings to judge current valuation levels.
- **DCF (Discounted Cash Flow)**:
  - Automated WACC (Weighted Average Cost of Capital) calculation module.
  - Flexible assumptions for growth rates and terminal value.
  - Supports TTM (Trailing Twelve Months) and FY (Fiscal Year) based projections.
- **Advanced Models**:
  - PEG Ratio analysis.
  - ROIC / ROE quality assessment.
- **Analyst Consensus**: Track and compare market expectations.

### 4. Interactive Visualization 📈

- Dynamic charts powered by Plotly.
- Analyze Revenue, Net Income, Margins, and Cash Flow trends over time.

---

## 🛠️ Architecture & Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) - For a responsive, interactive web UI.
- **Data Processing**: [Pandas](https://pandas.pydata.org/) - Heavy lifting of financial data manipulations.
- **Storage**: [SQLite](https://www.sqlite.org/) - Lightweight, local, serverless database for persistent storage.
- **External Data**: [yfinance](https://github.com/ranarousset/yfinance) - For real-time market data.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Seanyim/Stock-Ward.git
   cd Stock-Ward
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run main.py
   ```

4. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`.

---

## 📖 Usage Guide

### 1. Add a Company

- Navigate to the **Sidebar**.
- Enter the **Ticker** (e.g., `AAPL`, `600519.SS`, `9988.HK`).
- Select the **Region** and **Unit**.
- Click **"Add/Update Company"**.

### 2. Enter Financial Data

- Go to the **"📝 Data Entry"** tab.
- Input historical financial data (Revenue, Net Income, Cash Flows).
- The system will automatically fetch relevant stock prices if configured.

### 3. Analyze Trends

- Switch to **"📈 Trend Analysis"** to view growth rates and margins.
- Visualizations help identify cyclicality and stability.

### 4. Perform Valuation

- Use **"🧮 Valuation Models"** to estimate fair value.
- Adjust parameters in **DCF** or **PE** tabs to test different scenarios.

---

## ⚙️ Configuration

- **Proxy**: If you are in a region with restricted access to Yahoo Finance, you can configure a proxy URL in the sidebar (default: `http://127.0.0.1:10808`).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
