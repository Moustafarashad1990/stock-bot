import discord
from discord.ext import commands
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import traceback
import concurrent.futures
import torch
import torch.nn as nn
from polygon import RESTClient
import yfinance as yf  # News fallback
import os
from textblob import TextBlob

# ===================== CONFIG =====================
tickers = [
    "A", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP",
    "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMD", "AME", "AMGN",
    "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APP", "ARE", "ATO", "AVB",
    "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN",
    "BG", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA",
    "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG",
    "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS",
    "CNC", "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO",
    "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CVS", "CVX", "CZR", "D", "DAL", "DASH", "DAY", "DE",
    "DECK", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DPZ", "DRI", "DTE",
    "DUK", "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR",
    "ENPH", "EOG", "EPAM", "EQIX", "EQR", "ES", "ESS", "ETN", "ETR", "EVRG", "EW", "EXC", "EXPD",
    "EXPE", "EXR", "F", "FANG", "FAST", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FOX",
    "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GE", "GEHC", "GEN", "GILD", "GIS", "GL", "GLW", "GM",
    "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HES",
    "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM",
    "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV",
    "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM",
    "KDP", "KEY", "KEYS", "KHC", "KIM", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH",
    "LHX", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV",
    "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "MGM",
    "MKC", "MKTX", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI",
    "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NFLX", "NI", "NKE", "NOC",
    "NOW", "NRG", "NSC", "NTAP", "NTRS", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC",
    "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP",
    "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNR", "PNW", "PODD", "POOL", "PPL",
    "PRU", "PSA", "PSX", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RJF", "RL",
    "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVMD", "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SLB",
    "SMCI", "SNA", "SNPS", "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT",
    "TJX", "TKO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO",
    "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VICI", "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA",
    "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB", "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL",
    "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS", "ARES", "CRH", "CVNA", "FIX"
]

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "1452952904980758571"))
DATA_PERIOD_DAYS = 400
MAX_WORKERS = 30  # For paid Polygon

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY required!")

# ===================== LSTM MODEL =====================
class FastLSTM(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cpu")

# ===================== INDICATORS =====================
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def detect_ma_crossover(df: pd.DataFrame):
    if len(df) < 200:
        return None
    ma50 = df['close'].rolling(50).mean()
    ma200 = df['close'].rolling(200).mean()
    prev50, curr50 = ma50.iloc[-2], ma50.iloc[-1]
    prev200, curr200 = ma200.iloc[-2], ma200.iloc[-1]
    if prev50 <= prev200 and curr50 > curr200:
        return "Golden Cross"
    if prev50 >= prev200 and curr50 < curr200:
        return "Death Cross"
    return None

# ===================== LSTM PREDICTION =====================
def lstm_predict_next_close(close_prices: pd.Series) -> float | None:
    try:
        if len(close_prices) < 70:
            return None
        prices = close_prices[-120:].values.astype(np.float32)
        min_val, max_val = prices.min(), prices.max()
        scaled = (prices - min_val) / (max_val - min_val + 1e-7)
        lookback = 30
        if len(scaled) <= lookback:
            return None
        X = np.array([scaled[i:i + lookback] for i in range(len(scaled) - lookback)])
        y = scaled[lookback:]
        X_tensor = torch.from_numpy(X).unsqueeze(-1).to(device)
        y_tensor = torch.from_numpy(y).unsqueeze(-1).to(device)

        model = FastLSTM().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        last_seq = torch.from_numpy(scaled[-lookback:]).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred_scaled = model(last_seq).item()
        pred_price = pred_scaled * (max_val - min_val) + min_val
        return round(pred_price, 2)
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return None

# ===================== POLYGON DATA FETCH =====================
def fetch_polygon_data(client: RESTClient, ticker: str) -> pd.DataFrame | None:
    try:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=DATA_PERIOD_DAYS)
        aggs = list(client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.date(),
            to=end_date.date(),
            adjusted=True,
            limit=50000
        ))
        if not aggs:
            return None
        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df
    except Exception as e:
        print(f"Polygon fetch error for {ticker}: {e}")
        return None

# ===================== NEWS FUNCTION =====================
def get_stock_news(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        if not news:
            return "No recent news."
        msgs = []
        for item in news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '')
            sentiment = TextBlob(title).sentiment.polarity
            sent_str = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
            msgs.append(f"{title} ({publisher}, {sent_str})\n{link}")
        return "\n\n".join(msgs)
    except Exception as e:
        print(f"News fetch error: {e}")
        return "Failed to fetch news."

# ===================== TICKER PROCESSING =====================
def process_ticker(ticker: str, client: RESTClient) -> list[str]:
    df = fetch_polygon_data(client, ticker)
    if df is None or len(df) < 200:
        return []
    df = df.copy()
    df['close'] = df['Close']
    signals = []
    rsi_val = calculate_rsi(df['close']).iloc[-1]
    if rsi_val > 70:
        signals.append(f"RSI Overbought ({rsi_val:.1f})")
    if rsi_val < 30:
        signals.append(f"RSI Oversold ({rsi_val:.1f})")
    macd, signal = calculate_macd(df['close'])
    if len(macd) >= 2:
        if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
            signals.append("MACD Bullish Cross")
        if macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
            signals.append("MACD Bearish Cross")
    ma_sig = detect_ma_crossover(df)
    if ma_sig:
        signals.append(ma_sig)
    if signals:
        pred = lstm_predict_next_close(df['close'])
        if pred is not None:
            curr = df['close'].iloc[-1]
            change = pred - curr
            direction = "↑" if change > 0 else "↓"
            signals.append(f"LSTM → ${pred} ({direction}{abs(change):.2f})")
    return [f"{ticker}: {s}" for s in signals]

# ===================== DISCORD BOT =====================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command(name="scan")
async def manual_scan(ctx):
    await ctx.send(f"Manual scan triggered by {ctx.author.name}! Running analysis now...")
    await daily_analysis()

@bot.command(name="buy")
async def buy_recommendations(ctx):
    await ctx.send(f"Fetching buy recommendations for {ctx.author.name}...")
    await generate_recommendations(ctx, is_buy=True)

@bot.command(name="sell")
async def sell_recommendations(ctx):
    await ctx.send(f"Fetching sell recommendations for {ctx.author.name}...")
    await generate_recommendations(ctx, is_buy=False)

@bot.command(name="news")
async def stock_news(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("Usage: !news NVDA")
        return
    ticker = ticker.upper()
    await ctx.send(f"Getting news for {ticker}...")
    news = get_stock_news(ticker)
    await ctx.send(news)

async def generate_recommendations(ctx, is_buy: bool):
    channel = ctx.channel
    now = datetime.now(timezone.utc)
    title = "Buy" if is_buy else "Sell"
    await channel.send(f"Starting {title} recommendation scan - {now.strftime('%Y-%m-%d %H:%M')}")
    client = RESTClient(POLYGON_API_KEY)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t, client) for t in tickers]
        all_signals = []
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())
    client.close()
    recommendations = []
    if is_buy:
        for sig in all_signals:
            if "MACD Bullish Cross" in sig or "Golden Cross" in sig:
                recommendations.append(sig)
    else:
        for sig in all_signals:
            if "MACD Bearish Cross" in sig or "Death Cross" in sig or "RSI Overbought" in sig:
                recommendations.append(sig)
    if recommendations:
        msg = f"{title} Recommendations:\n" + "\n".join(recommendations[:30])
        if len(recommendations) > 30:
            msg += f"\n... +{len(recommendations) - 30} more"
        await channel.send(msg)
    else:
        await channel.send(f"No strong {title.lower()} recommendations today.")

async def daily_analysis():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found")
        return
    now = datetime.now(timezone.utc)
    await channel.send(f"Starting daily scan - {now.strftime('%Y-%m-%d %H:%M')}")
    client = RESTClient(POLYGON_API_KEY)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t, client) for t in tickers]
        all_signals = []
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())
    client.close()
    if all_signals:
        priority = [s for s in all_signals if "LSTM" in s or "Cross" in s]
        others = [s for s in all_signals if s not in priority]
        sorted_signals = priority + others
        msg = "High-Confidence Signals:\n" + "\n".join(sorted_signals[:30])
        if len(sorted_signals) > 30:
            msg += f"\n... +{len(sorted_signals) - 30} more"
        await channel.send(msg)
    else:
        await channel.send("No strong signals today.")

@bot.event
async def on_ready():
    print(f"Bot ready: {bot.user}")
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("Stock Analyzer with LSTM is online! Hourly scans during Dubai market hours ⚡")

async def schedule_daily():
    while True:
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        if weekday in [6, 0, 1, 2, 3] and hour in [6, 7, 8, 9, 10]:  # Dubai: Sun-Thu, 06:00-11:00 UTC
            await daily_analysis()
        await asyncio.sleep(60)  # Check every minute for precision

async def main():
    await asyncio.gather(
        bot.start(DISCORD_TOKEN),
        schedule_daily()
    )

if __name__ == "__main__":
    asyncio.run(main())