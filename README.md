# 美股量化半自动助手

一个基于 Streamlit 的本地化美股研究与半自动执行工作台，用公开数据把市场状态、持仓变化、期权结构、风险门控、样本外验证和研究摘要压缩成可读、可复核、可执行的分析结果。

> 这不是自动交易系统。
>
> 系统负责生成研究结论、风险门槛和执行参考，最终是否下单、如何下单、是否承担风险，始终由使用者自己决定。

## 项目定位

很多个人投资工作流最大的问题，不是没有信号，而是数据和判断分散在太多地方：

1. 行情、新闻、宏观、期权、借券、账户预算分散在不同页面。
2. 即使看到了机会，也很难把信号、风险和资金约束统一到同一张执行清单上。
3. 很多工具只给“方向”，不告诉你在当前保证金、借券条件和波动环境下这笔交易到底该不该做。

这个项目的目标不是“预测市场”，而是把这些离散输入整合成一套本地可运行、中文可读、适合人工复核的半自动分析系统。

## 核心能力

- 趋势过滤 RSI：只有在 RSI14 偏弱且长期趋势未破坏时才允许低吸，避免纯超卖条件下的接飞刀。
- 数据熔断网关：对历史行情、实时价格和成本价做异常值校验，防止坏点直接污染因子、盈亏和执行单。
- 市场状态机：综合 VIX、10Y 利率、ETF 活跃度代理、QQQ、NVDA、AAOI 状态生成风险偏好分数。
- 多空持仓跟踪：同时支持多头和空头持仓，动态计算可用预算、空头保证金和借券门控。
- 数学增强：提供 SVD 正交化因子权重、PCA 残差均值回归、图谱套利候选和学术 Alpha 摘要。
- 样本外研究：内置 Walk-Forward 回测和稳健性验证，避免只看单次样本内表现。
- 报告输出：支持页面预览、Markdown 报告、JSON 执行结构和快照归档。

## 期权模块升级

当前版本的期权模块已经从“成交量摘要 + 启发式建议”升级成“近月 ATM 链 + 自算 IV + IV Rank 驱动”的结构：

- 近月 ATM 链：不再只看最活跃执行价，而是抓取首个 7 到 45 DTE 的近月到期日，并围绕现价构建完整 ATM 窗口 Call / Put 链。
- 专业定价与隐波反解：使用 Black-Scholes-Merton、Vega、Newton 迭代和 bracket fallback 反算隐含波动率。
- 52 周 IV 本地缓存：每天把 ATM IV 写入本地缓存，逐步形成真实 IV 历史序列，供 IV Rank 直接使用。
- IVR 驱动策略选择：
  - IVR > 80：只建议卖波动结构，如 Short Strangle、Iron Condor。
  - IVR < 20：优先建议买波动结构，如 Long Straddle。
  - 中间区间：按持仓、趋势破坏和事件风险在 Covered Call、Bear Put Spread、Iron Condor 等结构间切换。
- 动态盈亏图：横轴按 $S_0 \cdot e^{\pm IV\sqrt{T}}$ 自动缩放，不再使用固定价格区间。
- 香港盈立证券步骤：页面内直接给出策略级别的下单顺序、检查项和成交后管理建议。

> 首次运行时，52 周 IV 缓存样本会比较少，此时 IVR 会暂时回退到代理带宽逻辑；缓存随着每日运行持续积累后，会逐步转为真实历史分位驱动。

## 界面预览

### 首页预览

![首页预览](docs/assets/homepage-overview.png)

### 样例报告预览

![样例报告预览](docs/assets/report-preview.png)

## 架构概览

```mermaid
flowchart LR
    input[持仓与账户输入]
    app[Streamlit 应用层]
    fetcher[数据抓取层\nyfinance / Eastmoney / IBorrowDesk]
    sanity[数据校验与熔断]
    engine[策略与数学引擎\nSignals / Kelly / Walk-Forward / Options / SVD-PCA / Graph Arb]
    report[输出层\n核心信号 / 持仓执行 / 期权预言机 / 异动雷达 / 报告]
    user[人工复核与下单]

    input --> app
    app --> fetcher
    fetcher --> sanity
    sanity --> engine
    app --> engine
    engine --> report
    report --> user
```

## 当前主要模块

### 市场与现货

- 核心指数信号：SPY、QQQ、DIA、IWM。
- 市场状态辅助标的：QQQ、NVDA、AAOI。
- 趋势过滤现货逻辑：结合 RSI、长期均线、波动率和风险门槛输出 ADD / HOLD / REDUCE / OBSERVE。

### 账户与风控

- 账户总权益、闲余资金、多空敞口、浮动盈亏与已实现盈亏拆分展示。
- 空头借券费率、可借股数、HTB 门控、动态初始保证金和维持担保比例。
- 成本价熔断和报价熔断，防止脏数据污染执行单。

### 数学增强

- SVD 正交化：识别当前横截面因子权重。
- PCA 残差均值回归：识别特质收益偏离最明显的标的。
- 图谱套利：扫描杠杆 ETF、主题 ETF、跨市场映射和先行滞后链路。
- 学术 Alpha 摘要：抓取近期研究主题并回灌为可执行因子草图与风控提示。

### 期权流与期权预言机

- 期权流摘要：展示近月期权成交概况、ATM 链、当前 ATM IV 和本地 52 周 IV 样本数。
- 期权预言机：对核心标的和持仓标的生成策略建议、盈亏图和下单步骤。
- 当前支持的主要结构：Covered Call、Bear Put Spread、Long Straddle、Short Strangle、Iron Condor。

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
.venv/bin/python -m streamlit run app.py --server.port 8501
```

浏览器访问：

```text
http://localhost:8501
```

## 最小使用流程

1. 在左侧录入持仓、方向、股数、成本价与备注。
2. 填写账户总权益、空头基准保证金和单笔仓位上限。
3. 调整凯利折扣，控制整体风险暴露。
4. 点击运行分析。
5. 先看“核心信号”和“市场状态”，再看“持仓盈亏”和“异动雷达”。
6. 若需要期权结构，进入“期权预言机”选择标的和策略，查看盈亏图与下单步骤。
7. 保存 Markdown / JSON 报告到 snapshots 目录做留档。

## 示例持仓模板

```json
[
  {
    "ticker": "NVDA",
    "side": "LONG",
    "shares": 10,
    "cost_basis": 191.2,
    "notes": "半导体/AI算力"
  },
  {
    "ticker": "QQQ",
    "side": "LONG",
    "shares": 2,
    "cost_basis": 520.0,
    "notes": "科技指数 ETF"
  }
]
```

## 项目结构

```text
.
├── app.py
├── config.py
├── alpha_researcher.py
├── arbitrage_scanner.py
├── math_engine.py
├── data/
├── models/
├── portfolio/
├── reports/
├── snapshots/
├── docs/
├── PRINCIPLES.md
├── USAGE.md
└── requirements.txt
```

## 关键文件

- app.py：Streamlit 主入口。
- data/fetcher.py：行情、扩展时段、新闻和数据源回退逻辑。
- data/options.py：近月 ATM 期权链、自算 IV 与本地 52 周 IV 缓存。
- data/sanity.py：历史行情、实时报价和成本价熔断网关。
- models/signals.py：核心信号与门槛逻辑。
- models/options_advisor.py：Black-Scholes、IV 反解、IVR 策略选择与盈亏图。
- math_engine.py：SVD / PCA 数学增强引擎。
- arbitrage_scanner.py：图谱套利与先行滞后扫描。
- walk_forward_validation.py：样本外验证与滚动研究。

## 数据源说明

- 行情：yfinance 为主，东方财富为回退。
- 借券：IBorrowDesk。
- 宏观：VIX、10Y Treasury、ETF 活跃度代理。
- 新闻：yfinance 新闻接口。
- 学术研究：arXiv q-fin、NBER 工作论文页面。

## 已知限制

1. 公开数据可能延迟、缺失、错配或阶段性失真。
2. 近月期权链在极短到期日和低流动性执行价上噪声很大，因此系统默认跳过过短到期并只围绕 ATM 窗口计算 IV。
3. 52 周 IV 缓存依赖本地持续运行逐步积累，不是外部付费历史 IV 数据库。
4. 系统生成的是研究与执行参考，不是自动成交引擎。

## 风险免责声明

1. 本项目仅用于研究、学习、界面展示与流程辅助，不构成投资建议。
2. 所有数据均来自公开源，可能存在延迟、缺失、错配或异常值。
3. 期权、杠杆和空头交易风险较高，实盘前应自行确认权限、保证金、流动性与券商下单细节。
4. 页面显示通过门控，不代表真实市场成交条件、滑点与尾部风险已经被完全覆盖。

## 文档

- [原理说明](PRINCIPLES.md)
- [使用说明](USAGE.md)
