from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_DIR / "snapshots"
HOLDINGS_FILE = BASE_DIR / "portfolio" / "holdings.json"

CACHE_TTL_SECONDS = 300
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04
DEFAULT_KELLY_FRACTION = 0.5

CORE_SIGNAL_TICKERS = ["SPY", "QQQ", "DIA", "IWM"]
REGIME_SUPPORT_TICKERS = ["QQQ", "NVDA", "AAOI"]
CORE_TICKERS = REGIME_SUPPORT_TICKERS
DEFAULT_HOLDING_TICKER = "NVDA"
MACRO_TICKERS = {
    "us10y": "^TNX",
    "vix": "^VIX",
}
OPTIONS_TICKER = "NVDA"
AAOX_UNDERLYING = "AAOI"

# 免费数据源下，ETF 资金净流入难以稳定获取，这里用活跃度代理变量替代。
ETF_FLOW_PROXY_TICKERS = ["QQQ", "SPY", "IVV"]

RADAR_UNIVERSE = [
    "AAPL",
    "AMZN",
    "AMD",
    "AVGO",
    "META",
    "MSFT",
    "NFLX",
    "TSLA",
    "PLTR",
    "SMCI",
    "ARM",
    "MU",
    "ASML",
    "CRWD",
    "PANW",
    "NET",
    "SNOW",
    "APP",
    "TTD",
    "UBER",
    "HOOD",
    "RDDT",
    "COIN",
    "MSTR",
    "SHOP",
    "INTU",
    "ADBE",
    "INTC",
    "ORCL",
    "CRM",
    "ANET",
    "MRVL",
    "LRCX",
    "KLAC",
    "CIEN",
    "GLW",
    "AAOI",
    "NBIS",
    "SOUN",
    "IONQ",
]

TICKER_METADATA = {
    "SPY": {"name": "标普500 ETF", "category": "大盘指数ETF"},
    "QQQ": {"name": "纳斯达克100 ETF", "category": "科技成长指数ETF"},
    "DIA": {"name": "道琼斯工业平均 ETF", "category": "蓝筹指数ETF"},
    "IWM": {"name": "罗素2000 ETF", "category": "中小盘指数ETF"},
    "NVDA": {"name": "英伟达", "category": "半导体/AI算力"},
    "AAOI": {"name": "Applied Optoelectronics", "category": "光模块/CPO"},
    "AAOX": {"name": "2倍做多光模块", "category": "杠杆ETF/光模块"},
    "AAPL": {"name": "苹果", "category": "科技股/消费电子"},
    "AMZN": {"name": "亚马逊", "category": "科技股/电商云"},
    "AMD": {"name": "超威半导体", "category": "半导体/CPU GPU"},
    "AVGO": {"name": "博通", "category": "半导体/网络芯片"},
    "META": {"name": "Meta", "category": "科技股/互联网广告"},
    "MSFT": {"name": "微软", "category": "科技股/云软件"},
    "NFLX": {"name": "奈飞", "category": "流媒体/互联网"},
    "TSLA": {"name": "特斯拉", "category": "新能源车"},
    "PLTR": {"name": "Palantir", "category": "软件/数据分析"},
    "SMCI": {"name": "超微电脑", "category": "服务器/AI硬件"},
    "ARM": {"name": "Arm", "category": "半导体/IP授权"},
    "MU": {"name": "美光", "category": "存储板块/半导体"},
    "ASML": {"name": "阿斯麦", "category": "半导体设备"},
    "CRWD": {"name": "CrowdStrike", "category": "网络安全"},
    "PANW": {"name": "Palo Alto Networks", "category": "网络安全"},
    "NET": {"name": "Cloudflare", "category": "云网络/安全"},
    "SNOW": {"name": "Snowflake", "category": "云数据"},
    "APP": {"name": "AppLovin", "category": "广告科技"},
    "TTD": {"name": "Trade Desk", "category": "广告科技"},
    "UBER": {"name": "优步", "category": "互联网出行"},
    "HOOD": {"name": "Robinhood", "category": "互联网券商/金融科技"},
    "RDDT": {"name": "Reddit", "category": "互联网社交"},
    "COIN": {"name": "Coinbase", "category": "加密交易所/金融科技"},
    "MSTR": {"name": "MicroStrategy", "category": "比特币概念/软件"},
    "SHOP": {"name": "Shopify", "category": "电商SaaS"},
    "INTU": {"name": "Intuit", "category": "金融软件"},
    "ADBE": {"name": "Adobe", "category": "创意软件"},
    "INTC": {"name": "英特尔", "category": "半导体/CPU"},
    "ORCL": {"name": "甲骨文", "category": "企业软件/云"},
    "CRM": {"name": "Salesforce", "category": "企业软件/SaaS"},
    "ANET": {"name": "Arista", "category": "网络设备/AI互联"},
    "MRVL": {"name": "迈威尔", "category": "半导体/网络芯片"},
    "LRCX": {"name": "Lam Research", "category": "半导体设备"},
    "KLAC": {"name": "科磊", "category": "半导体设备"},
    "CIEN": {"name": "Ciena", "category": "光通信/网络设备"},
    "GLW": {"name": "康宁", "category": "光学材料/玻璃"},
    "NBIS": {"name": "Nebius", "category": "AI基础设施"},
    "SOUN": {"name": "SoundHound", "category": "语音AI"},
    "IONQ": {"name": "IonQ", "category": "量子计算"},
}

TICKER_SUGGESTION_UNIVERSE = sorted(
    set(CORE_SIGNAL_TICKERS + REGIME_SUPPORT_TICKERS + RADAR_UNIVERSE + ["AAOX"]) 
)

POSITIVE_KEYWORDS = {
    "beat": 0.8,
    "beats": 0.8,
    "surge": 0.7,
    "rally": 0.6,
    "upgrade": 0.7,
    "strong": 0.5,
    "growth": 0.5,
    "guidance raise": 0.9,
    "expansion": 0.5,
    "partnership": 0.4,
    "ai": 0.3,
    "record": 0.5,
    "approval": 0.7,
}

NEGATIVE_KEYWORDS = {
    "miss": -0.8,
    "misses": -0.8,
    "downgrade": -0.7,
    "lawsuit": -0.7,
    "fraud": -1.0,
    "delay": -0.4,
    "weak": -0.5,
    "cuts guidance": -0.9,
    "investigation": -0.8,
    "selloff": -0.6,
    "decline": -0.5,
    "tariff": -0.3,
    "war": -0.4,
    "recall": -0.6,
}
