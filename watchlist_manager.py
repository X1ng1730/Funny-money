from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
WATCHLIST_PATH = DATA_DIR / "watchlist_master.csv"
CATEGORIES_PATH = DATA_DIR / "categories.csv"

WATCHLIST_COLUMNS = [
    "ticker",
    "category",
    "quick_thesis",
    "macro_tag",
    "manual_catalyst",
    "priority",
    "active",
]
CATEGORY_COLUMNS = ["category", "description"]

DEFAULT_CATEGORIES = {
    "Open Positions": "Current open swing positions.",
    "Weekly Watchlist / High-Priority Movers": "Near-term stocks and ETFs to scan closely this week.",
    "AI Compute & Core Semiconductors": "Core compute, silicon, and semiconductor leaders.",
    "AI Memory, Storage & Data Throughput": "Memory, storage, and throughput infrastructure.",
    "Semiconductor Equipment, Test, Packaging & Fab Support": "Semiconductor supply chain support names.",
    "AI Data-Center Infrastructure, Networking, Power & Optical Support": "Data-center infrastructure, power, networking, and optical support.",
    "Cloud, Enterprise Software, Cybersecurity & AI Workflow": "Software, cloud, security, and AI workflow names.",
    "Internet, E-Commerce, Streaming & Digital Ads": "Internet platforms, commerce, streaming, and advertising.",
    "Space, Satellites, Launch & Commercial Space": "Space launch, satellites, and commercial space exposure.",
    "Aerospace, Defense & Aviation Suppliers": "Defense, aviation, and aerospace suppliers.",
    "Airlines, Travel & Oil-Sensitive Cyclicals": "Airlines, travel demand, and oil-sensitive cyclicals.",
    "Energy, Power, Infrastructure & Heavy Industry": "Energy, power, infrastructure, industrial, and heavy industry names.",
    "Consumer Growth, Restaurants, Retail & Lifestyle": "Consumer growth, restaurant, retail, and lifestyle names.",
    "Financials, Fintech & Credit": "Financials, fintech, credit, and lending names.",
    "Healthcare, Pharma & Biotech": "Healthcare, pharma, and biotech names.",
}

DEFAULT_TICKERS_BY_CATEGORY = {
    "Open Positions": ["SOFI", "APLD", "MU", "NASA", "SHOP", "GOOGL", "META"],
    "Weekly Watchlist / High-Priority Movers": ["SPCX", "MRVL", "YOU", "INTC", "LUV", "IONQ", "RKLB", "DAL", "UAL", "NFLX", "RDW"],
    "AI Compute & Core Semiconductors": ["NVDA", "AMD", "AVGO", "MRVL", "QCOM", "INTC", "TSM", "STM", "TXN"],
    "AI Memory, Storage & Data Throughput": ["MU", "DRAM", "WDC", "STX", "SNDK", "KIOXIA", "SIMO"],
    "Semiconductor Equipment, Test, Packaging & Fab Support": ["ACMR", "AEHR", "COHU", "ICHR", "UCTT", "AMKR", "VECO", "FORM", "ACLS", "AXTI"],
    "AI Data-Center Infrastructure, Networking, Power & Optical Support": ["APLD", "DELL", "JBL", "CIEN", "NOK", "TMUS", "CEG", "NRG", "AMSC", "POET"],
    "Cloud, Enterprise Software, Cybersecurity & AI Workflow": ["PANW", "CRWD", "FTNT", "PLTR", "SNOW", "ORCL", "CRM", "WDAY", "NOW", "SAP", "IBM", "MSFT", "ADBE", "DOCU", "ZM", "YOU"],
    "Internet, E-Commerce, Streaming & Digital Ads": ["GOOGL", "META", "AMZN", "SHOP", "NFLX", "SPOT", "FOX"],
    "Space, Satellites, Launch & Commercial Space": ["SPCX", "NASA", "RKLB", "RDW", "ASTS", "SPIR", "SPCE", "LUNR", "VOYG", "FLY"],
    "Aerospace, Defense & Aviation Suppliers": ["LMT", "MRCY", "ATROB", "FTAI"],
    "Airlines, Travel & Oil-Sensitive Cyclicals": ["DAL", "UAL", "AAL", "LUV", "USO", "XOM", "CVX", "BP"],
    "Energy, Power, Infrastructure & Heavy Industry": ["CAT", "STRL", "LYB", "NRG", "CEG", "JBL", "AMSC", "TSLA", "BP"],
    "Consumer Growth, Restaurants, Retail & Lifestyle": ["LULU", "CELH", "CAVA", "CMG", "COST", "CROX", "ULTA", "VSXY", "SFM", "KMX"],
    "Financials, Fintech & Credit": ["SOFI", "JPM", "BMO", "KMX"],
    "Healthcare, Pharma & Biotech": ["ABBV", "ADPT"],
}


def _default_watchlist_rows() -> list[dict]:
    rows = []
    examples = {
        "MRVL": ("AI networking + custom silicon", "AI capex", "Earnings / AI data-center commentary", "High"),
        "RKLB": ("Launch + space infrastructure", "Risk-on space", "Launch / index / contract news", "High"),
        "DAL": ("Airline demand + oil sensitivity", "Oil-sensitive", "Oil prices / travel demand", "Medium"),
    }
    for category, tickers in DEFAULT_TICKERS_BY_CATEGORY.items():
        for ticker in tickers:
            thesis, macro, catalyst, priority = examples.get(ticker, ("", "", "", "Medium"))
            if category == "Open Positions":
                priority = "High"
            rows.append(
                {
                    "ticker": ticker,
                    "category": category,
                    "quick_thesis": thesis,
                    "macro_tag": macro,
                    "manual_catalyst": catalyst,
                    "priority": priority,
                    "active": True,
                }
            )
    return rows


def ensure_watchlist_files() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    categories = pd.DataFrame(
        [{"category": category, "description": description} for category, description in DEFAULT_CATEGORIES.items()]
    )
    if CATEGORIES_PATH.exists():
        current = pd.read_csv(CATEGORIES_PATH)
        current = _normalize_categories(current)
        categories = pd.concat([current, categories], ignore_index=True)
        categories = categories.drop_duplicates(subset=["category"], keep="first")
    categories.to_csv(CATEGORIES_PATH, index=False)

    defaults = pd.DataFrame(_default_watchlist_rows())
    if WATCHLIST_PATH.exists():
        current = pd.read_csv(WATCHLIST_PATH)
        current = _normalize_watchlist(current)
        defaults = pd.concat([current, defaults], ignore_index=True)
    defaults = _normalize_watchlist(defaults)
    defaults = defaults.drop_duplicates(subset=["ticker", "category"], keep="first")
    defaults.to_csv(WATCHLIST_PATH, index=False)


def _normalize_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    for column in WATCHLIST_COLUMNS:
        if column not in df.columns:
            df[column] = True if column == "active" else ""
    df = df[WATCHLIST_COLUMNS].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["active"] = df["active"].map(_to_bool)
    for column in ["quick_thesis", "macro_tag", "manual_catalyst", "priority"]:
        df[column] = df[column].fillna("").astype(str)
    df = df[(df["ticker"] != "") & (df["category"] != "")]
    return df


def _normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    for column in CATEGORY_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[CATEGORY_COLUMNS].copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str)
    return df[df["category"] != ""]


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().upper() in {"TRUE", "1", "YES", "Y", "ACTIVE"}


def load_watchlist() -> pd.DataFrame:
    ensure_watchlist_files()
    return _normalize_watchlist(pd.read_csv(WATCHLIST_PATH))


def save_watchlist(df: pd.DataFrame) -> pd.DataFrame:
    ensure_watchlist_files()
    cleaned = _normalize_watchlist(df)
    cleaned = cleaned.drop_duplicates(subset=["ticker", "category"], keep="last")
    cleaned.to_csv(WATCHLIST_PATH, index=False)
    return cleaned


def load_categories() -> pd.DataFrame:
    ensure_watchlist_files()
    return _normalize_categories(pd.read_csv(CATEGORIES_PATH))


def save_categories(df: pd.DataFrame) -> pd.DataFrame:
    ensure_watchlist_files()
    cleaned = _normalize_categories(df)
    cleaned = cleaned.drop_duplicates(subset=["category"], keep="last")
    cleaned.to_csv(CATEGORIES_PATH, index=False)
    return cleaned


def add_category(category: str, description: str = "") -> pd.DataFrame:
    categories = load_categories()
    new_row = pd.DataFrame([{"category": category.strip(), "description": description.strip()}])
    return save_categories(pd.concat([categories, new_row], ignore_index=True))


def add_watchlist_row(
    ticker: str,
    category: str,
    quick_thesis: str = "",
    macro_tag: str = "",
    manual_catalyst: str = "",
    priority: str = "Medium",
    active: bool = True,
) -> pd.DataFrame:
    watchlist = load_watchlist()
    new_row = pd.DataFrame(
        [
            {
                "ticker": ticker.upper().strip(),
                "category": category.strip(),
                "quick_thesis": quick_thesis,
                "macro_tag": macro_tag,
                "manual_catalyst": manual_catalyst,
                "priority": priority,
                "active": active,
            }
        ]
    )
    return save_watchlist(pd.concat([watchlist, new_row], ignore_index=True))
