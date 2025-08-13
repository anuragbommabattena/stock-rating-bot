import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="US + India Rating Bot", page_icon="📈", layout="centered")

# ---------------------------
# Helpers
# ---------------------------
def zclip(x, lo=0.0, hi=10.0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    return max(lo, min(hi, float(x)))

def safe_div(a,b,default=np.nan):
    try:
        if b in (None,0): return default
        return a/b
    except Exception:
        return default

def renorm_weights(d):
    s = sum(v for v in d.values() if v is not None)
    return {k:(v/s if s>0 else 0) for k,v in d.items()}

# ---------------------------
# Detection
# ---------------------------
def detect_country_and_type(code_raw: str):
    s = code_raw.strip().upper()
    if s.startswith("INF") or (len(s)==5 and s.endswith("X")):
        return ("India" if s.startswith("INF") else "US", "Mutual Fund", s)
    if s.startswith("NSE:") or s.startswith("BSE:"):
        return ("India", "Stock", s)
    if s.endswith("BEES"):
        return ("India", "ETF", s)
    return ("US", "Stock", s)

# ---------------------------
# Data
# ---------------------------
SECTOR_MEDIANS = {
    "US": {
        "Technology": {"PE": 28, "EVEBITDA": 19, "PS": 6.0, "PFCF": 25},
        "Default":    {"PE": 18, "EVEBITDA": 12, "PS": 2.0, "PFCF": 18},
    },
    "India": {
        "Technology": {"PE": 30, "EVEBITDA": 20, "PS": 7.0, "PFCF": 28},
        "Default":    {"PE": 22, "EVEBITDA": 14, "PS": 3.0, "PFCF": 22},
    }
}

CATEGORY_MEDIANS = {
    "US": {
        "US Large Cap Blend": {"Expense": 0.05, "Return1Y": 0.22, "Return3Y": 0.11, "Volatility": 0.18, "Yield": 0.013},
        "Default":            {"Expense": 0.20, "Return1Y": 0.14, "Return3Y": 0.08, "Volatility": 0.22, "Yield": 0.02},
    },
    "India": {
        "India Equity Multi Cap": {"Expense": 0.011, "Return1Y": 0.28, "Return3Y": 0.15, "Volatility": 0.23, "Yield": 0.008},
        "Default":                {"Expense": 0.015, "Return1Y": 0.20, "Return3Y": 0.11, "Volatility": 0.25, "Yield": 0.01},
    }
}

UNIVERSE = {
    "US_STOCKS": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META"],
    "IN_STOCKS": ["NSE:INFY","NSE:TCS","NSE:RELIANCE","NSE:HDFCBANK","NSE:ITC","NSE:LTIM"],
    "US_ETFS": ["VOO","VTI","QQQ","IWM"],
    "IN_ETFS": ["NSE:NIFTYBEES","NSE:BANKBEES","NSE:JUNIORBEES"],
    "US_MF": ["VFIAX","FXAIX","SWPPX"],
    "IN_MF": ["INF109K01ZB3"]
}

def fetch_yahoo(ticker: str):
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    hist = tk.history(period="3y", interval="1d")
    price = float(info.get("currentPrice") or info.get("regularMarketPrice") or (hist["Close"].iloc[-1] if not hist.empty else np.nan))
    sector = info.get("sector")
    industry = info.get("industry")
    beta = info.get("beta")
    pe = info.get("trailingPE")
    pb = info.get("priceToBook")
    ps = info.get("priceToSalesTrailing12Months")
    pfcf = info.get("trailingPegRatio")
    div_yield = info.get("dividendYield")
    r1 = r3 = np.nan
    if not hist.empty:
        c = hist["Close"]
        try:
            if len(c) > 252: r1 = float(c.iloc[-1]/c.iloc[-252]-1)
            r3 = float(c.iloc[-1]/c.iloc[0]-1)
        except Exception:
            pass
    return dict(price=price, sector=sector, industry=industry, beta=beta, pe=pe, pb=pb, ps=ps, pfcf=pfcf, div_yield=div_yield, r_1y=r1, r_3y=r3)

def fetch_for(code: str, country: str, itype: str):
    if country=="India":
        core = code.replace("NSE:","").replace("BSE:","")
        data = fetch_yahoo(core + ".NS")
        if np.isnan(data["price"]):
            data = fetch_yahoo(core + ".BO")
        return data
    else:
        return fetch_yahoo(code)

# ---------------------------
# Factor models & weights
# ---------------------------
STOCK_W = renorm_weights({
    "pe_vs_sector": 0.125, "financial_health": 0.125, "valuation": 0.125, "growth": 0.125,
    "management": 0.125, "moat": 0.125, "risk": 0.125, "dividend": 0.125
})
ETF_W = renorm_weights({
    "expense": 0.15, "liquidity": 0.10, "holdings_quality": 0.20, "sector_strength": 0.15,
    "valuation_vs_bench": 0.10, "performance": 0.15, "volatility": 0.10, "yield_stability": 0.05
})
MF_W = renorm_weights({
    "expense": 0.10, "aum_size": 0.10, "manager_track": 0.15, "cat_rel_perf": 0.15,
    "risk_adj": 0.15, "consistency": 0.15, "downside": 0.10, "portfolio_quality": 0.10
})

def score_stock(country, tkr, m):
    sector = m.get("sector") or "Default"
    med = SECTOR_MEDIANS.get(country, {}).get(sector, SECTOR_MEDIANS[country]["Default"])
    exps, scores = {}, {}

    pe = m.get("pe")
    scores["pe_vs_sector"] = zclip(5 + 5*((med["PE"]/pe)-1)) if (pe and med["PE"]) else np.nan
    exps["pe_vs_sector"] = f"P/E {pe} vs sector {med['PE']}"

    pb = m.get("pb")
    scores["financial_health"] = zclip(10 - abs((pb or 3)-3)*2) if pb else np.nan
    exps["financial_health"] = f"P/B {pb} (target ~3)"

    ps, pfcf = m.get("ps"), m.get("pfcf")
    comp=[]
    if pe and med["PE"]: comp.append(5+5*(med["PE"]/pe-1))
    if ps and med["PS"]: comp.append(5+5*(med["PS"]/ps-1))
    if pfcf and med["PFCF"]: comp.append(5+5*(med["PFCF"]/pfcf-1))
    scores["valuation"] = zclip(np.nanmean(comp) if comp else np.nan)
    exps["valuation"] = "PE/PS/PFCF vs sector medians"

    r1, r3 = m.get("r_1y"), m.get("r_3y")
    scores["growth"] = zclip(5 + np.nanmean([(r1 or 0)*10, (r3 or 0)*5]))
    exps["growth"] = f"1Y {r1:.1%} / 3Y {r3:.1%}"

    scores["management"] = 6.0; exps["management"]="Proxy (v1=6)"
    scores["moat"] = 6.0; exps["moat"]="Proxy (v1=6)"

    beta = m.get("beta") or 1.0
    scores["risk"] = zclip(10 - abs(beta-1)*5); exps["risk"]=f"Beta {beta}"

    y = m.get("div_yield")
    scores["dividend"] = zclip((y*100/0.6) if y not in (None, np.nan) else np.nan)
    exps["dividend"] = f"Yield {y:.2%}" if y else "No dividend"

    w = renorm_weights({k:(STOCK_W[k] if not np.isnan(scores[k]) else 0) for k in STOCK_W})
    overall = sum(w[k]*scores[k] for k in STOCK_W if not np.isnan(scores[k]))
    return round(overall,2), scores, exps, sector

def reco(overall, buy_thr=8.0, hold_thr=6.0):
    return "Buy" if overall>=buy_thr else ("Hold" if overall>=hold_thr else "Sell/Avoid")

# ---------------------------
# UI
# ---------------------------
st.title("📈 US + India: Stock, ETF & Mutual Fund Rating (AI Bot)")
st.caption("Educational use only. Not investment advice.")

col1, col2, col3 = st.columns(3)
with col1:
    us_ticker = st.text_input("US ticker (Stock/ETF): e.g., AAPL, VOO")
with col2:
    in_ticker = st.text_input("India ticker (Stock/ETF): e.g., NSE:INFY, NSE:NIFTYBEES")
with col3:
    mf_code = st.text_input("Mutual fund code/ISIN: e.g., VFIAX or INF109K01ZB3")

query = None
for candidate in [us_ticker, in_ticker, mf_code]:
    if candidate.strip():
        query = candidate.strip()
        break

buy_thr = st.slider("Buy threshold", 0.0, 10.0, 8.0, 0.5)
hold_thr = st.slider("Hold threshold", 0.0, 10.0, 6.0, 0.5)

if query:
    country, itype, code = detect_country_and_type(query)
    st.write(f"**Detected:** {country} · {itype} · `{code}`")

    data = fetch_for(code, country, itype)
    if np.isnan(data.get("price", np.nan)):
        st.error("Could not fetch data for this code/ticker.")
        st.stop()

    if itype=="Stock":
        overall, scores, exps, sector = score_stock(country, code, data)
        rec = reco(overall, buy_thr, hold_thr)
        st.subheader(f"Overall: {overall:.2f}/10 → {rec}")
        st.write(f"**Sector:** {sector or 'n/a'}")
    else:
        st.subheader("ETF/MF scoring not fully implemented in this trimmed demo.")

    df = pd.DataFrame({
        "Factor": list(scores.keys()),
        "Weight": [STOCK_W.get(k) for k in scores.keys()],
        "Score(0-10)": [scores[k] for k in scores.keys()],
        "Why": [exps[k] for k in exps.keys()]
    })
    st.dataframe(df, use_container_width=True)

    km = {k:v for k,v in data.items() if k in ["price","pe","pb","ps","pfcf","div_yield","beta","r_1y","r_3y","sector","industry"]}
    st.markdown("**Key metrics**")
    st.json(km)
else:
    st.info("Enter exactly one input: US ticker or India ticker or MF code/ISIN.")
