## 2025/12/29

### updates

- update ver 1.1
- update financial report with accumalted data (Q1, H1, Q9, FY)
- update calculator.py add YoY, TTM YoY, QoQ, single quarter YoY
- update PE, TTM PE etc.. by yoy, ttm growth_rate
- update DCF

### questionable

- DCF TTM calculation
- PE
- charts
- data_entry input FCF
- DCF tab use different wacc interest
- sync with git

## 2025/12/30

### updates

- update data_entry.py to scable function
- new add config.py to accept new added financial data
- update ttm period calculation
- update charts and yoy data looks
- upload local files to github repository
- update NVDA financial data

### questionable

- DCF
- wacc calculation
- PEG revise by fisher interest

## 2025/12/31

### updates

- review and update DCF & WACC
- update config financial data list
- update logic if financial data needs record

### questionable

- config needs shares & stock price calculation instead of market cap
- DCF, WACC, E/V, D/V check

## 2026/1/8

### updates

- clear FCF TTM model
- clear E/V, D/V use newest financial data not TTM calculation
- update README
- update venv for different enviroment
- update PE & PEG model with TTM

### questionable

- growth quality R
- monte carlo -> stock not options
- financial data need update

## 2026/1/12

### update

- create new branch 'fetch'
- 'fetch' allows fetch data from yfinance
- 'fetch' uses qslite as databases

## 2026/1/24

### update

- revise stock price fetch from yfinance
- fix bugs
- allow diverse financial data manually.

## 2026/2/24

### updates

- rewrite README.md and README_ZH.md with detailed descriptions and structure
- auto-generate screenshots for Glimpse Inside section using Playwright
- push updated README and screenshots to GitHub

## 2026/2/26

### updates (v2.2)

- **group management optimization**
  - auto-detect market region from ticker suffix (e.g. `.SS`/`.SZ` → CN, `.HK` → HK, `.T` → JP, `.TW` → TW)
  - company selector changed to two-level cascade: select group → select company within group
  - member management tab optimized: add shows only companies not in group, remove shows only group members
  - new db functions: `detect_region_from_ticker()`, `get_companies_in_category()`, `get_companies_not_in_category()`

- **auto unit detection for financial data**
  - removed manual unit selector (Billion/Million) from add-company form
  - unit auto-detected from region, all data stored as Billion internally
  - JSON import `parse_value()` now supports 亿/百万/万 per-value conversion to Billion
  - mixed-unit JSON data (e.g. revenue in 亿, expenses in 万) handled correctly

- 10 automated tests passing (region detection, category scoping, smoke)
