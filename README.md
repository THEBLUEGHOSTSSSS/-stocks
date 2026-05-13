# 美股量化半自动助手

一个基于 Streamlit 的本地化美股分析系统，用公开数据完成市场状态判断、持仓跟踪、执行参考、样本外回测、期权结构建议与报告输出。

这不是自动交易系统。

系统负责把行情、新闻、宏观、期权与借券信息压缩成可执行的分析结果，最终是否下单、如何下单，仍由你自己决定。

## 核心亮点

- 趋势过滤 RSI：现货侧已切换为 Trend-Filtered RSI，只有在 `RSI14 < 35` 且 `close > SMA120` 时才允许低吸，避免“接飞刀”。
- 持仓脏数据熔断：当持仓成本价与当前市价偏离超过 80% 时，运行时自动熔断并重置，防止盈亏和仓位计算失真。
- 夜盘/盘前扩展时段：支持盘前、盘中、夜盘价格抓取与中文展示。
- 期权策略预言机：可生成 Covered Call、Bear Put Spread、Strangle 建议，附带到期盈亏图与香港盈立证券具体下单步骤。
- 市场状态机：结合 VIX、10 年期美债、ETF 活跃度代理、QQQ/NVDA/AAOI 状态，判断风险偏好。
- 数学增强：包含 SVD 正交化因子权重、PCA 残差均值回归、图谱套利候选与学术 Alpha 摘要。
- 风控与资金管理：支持凯利折扣、波动率压制、空头借券约束、动态保证金与账户预算约束。
- 样本外研究：内置 walk-forward 回测与稳健性研究流程。

## 当前策略框架

当前系统的活跃现货逻辑不是追涨型短周期动量，而是更保守的趋势过滤均值回归框架：

1. `RSI14 < 35`：进入观察区。
2. `close > SMA120`：确认长期趋势未破坏，允许低吸。
3. `close <= SMA120`：判定为接飞刀风险，强制观望。
4. `RSI14 >= 70`：进入止盈/收缩区。
5. 仓位压制仍保留波动率公式：`clip(0.20 / hist_vol_20d, 0.10, 0.80)`。

## 功能概览

### 核心分析

- 核心指数信号：SPY、QQQ、DIA、IWM
- 市场状态判断：Risk-On / Risk-Off / Choppy / Range
- 持仓执行参考：ADD、HOLD、REDUCE、LIQUIDATE、ADD_SHORT、COVER
- 新 Alpha 候选：异动雷达、图谱套利、残差回归

### 数据与风控

- 海外行情优先使用 yfinance，异常或不可达时自动回退到东方财富
- 借券数据接入 IBorrowDesk，用于空头费率与可借股数约束
- 历史行情与实时行情都带有硬熔断 sanity gate
- 持仓成本价带运行时熔断，避免脏数据污染分析结果

### 期权模块

- 近月期权流摘要
- 期权策略预言机
- Plotly 到期盈亏图
- 香港盈立证券具体下单步骤

## 技术栈

- Python 3
- Streamlit
- pandas / numpy / scipy / statsmodels / scikit-learn
- yfinance
- plotly
- networkx

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd stocks
```

### 2. 创建虚拟环境

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动应用

```bash
streamlit run app.py --server.port 8501
```

打开浏览器访问：

```text
http://localhost:8501
```

## 使用流程

1. 在左侧栏录入持仓、方向、股数、成本价与备注。
2. 填写账户总权益、空头基准保证金和仓位上限。
3. 调整凯利折扣，控制整体风险暴露。
4. 点击“运行分析”。
5. 在页面查看：
   - 核心指数信号
   - 持仓盈亏与执行参考
   - 异动雷达
   - 样本外回测
   - 报告与 JSON
   - 原理与使用说明

## 页面模块

### 核心信号

- 市场状态与风险偏好
- 核心指数推演日志
- 期权流摘要
- 期权策略预言机
- 数学增强结果

### 持仓盈亏

- 账户总权益、闲余资金、多空敞口
- 当前持仓市值与浮盈亏
- 历史快照对比
- 持仓成本价熔断提示

### 异动雷达

- 放量突破 / 破位候选
- 图谱套利候选
- 新 Alpha 目标排序

### 样本外回测

- Walk-forward 回测
- 因子研究与稳健性比较

## 目录结构

```text
.
├── app.py
├── config.py
├── data/
├── models/
├── portfolio/
├── reports/
├── snapshots/
├── walk_forward_validation.py
├── PRINCIPLES.md
├── USAGE.md
└── requirements.txt
```

## 重点文件

- [app.py](app.py)：Streamlit 主入口
- [models/signals.py](models/signals.py)：核心信号引擎
- [models/options_advisor.py](models/options_advisor.py)：期权策略预言机与盈亏图
- [data/fetcher.py](data/fetcher.py)：行情、扩展时段与回退数据源抓取
- [data/indicators.py](data/indicators.py)：RSI、ATR、波动率、均线等指标
- [walk_forward_validation.py](walk_forward_validation.py)：样本外回测与稳健性研究
- [PRINCIPLES.md](PRINCIPLES.md)：原理说明
- [USAGE.md](USAGE.md)：使用说明

## 数据源说明

- 行情：yfinance 为主，东方财富为回退
- 借券：IBorrowDesk
- 宏观：VIX、10Y Treasury、ETF 活跃度代理
- 新闻：yfinance 新闻接口

## 适用场景

- 本地管理美股多头 / 空头持仓
- 需要半自动执行参考而不是全自动交易
- 想在一个界面里同时看信号、风控、期权建议、回测和报告

## 风险提示

1. 本项目不是投资建议。
2. 所有数据均来自公开源，可能延迟、缺失或异常。
3. 期权、杠杆和空头交易风险较高，实盘前请自行确认权限、保证金与下单细节。
4. 系统输出的是分析结果，不应被视为自动交易指令。

## 文档

- [原理说明](PRINCIPLES.md)
- [使用说明](USAGE.md)

## GitHub 展示建议

如果你准备公开放到 GitHub，建议至少补充以下内容：

1. 一张首页截图或 GIF。
2. 一份样例报告截图。
3. 一份示例持仓文件模板。
4. 你的 License 文件。

这样首页可读性会明显更好。
