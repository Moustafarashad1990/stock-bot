import discord
from discord.ext import commands
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import concurrent.futures
import os
from polygon import RESTClient
import yfinance as yf
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
MAX_WORKERS = 20  # Reduced slightly for safety

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY required!")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN required!")

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

# ===================== DATA FETCH =====================
def fetch_polygon_data(ticker: str) -> pd.DataFrame | None:
    try:
        client = RESTClient(POLYGON_API_KEY)  # New client per thread
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
        # Removed client.close() - not needed and causes errors
        
        if not aggs:
            print(f"No data for {ticker}")
            return None
            
        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'close', 'Volume']
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# ===================== NEWS =====================
def get_stock_news(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        if not news:
            return "No recent news found."
        msgs = []
        for item in news:
            title = item.get('title', 'No title')
            publisher = item.get('publisher', 'Unknown')
            link = item.get('link', '')
            sentiment = TextBlob(title).sentiment.polarity
            sent_str = "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
            msgs.append(f"• {title} ({publisher} | {sent_str})\n{link}")
        return "\n".join(msgs)
    except Exception as e:
        print(f"News fetch error for {ticker}: {e}")
        return "Failed to fetch news - try later."

# ===================== SIGNAL PROCESSING =====================
def process_ticker(ticker: str) -> list[str]:
    df = fetch_polygon_data(ticker)
    if df is None or len(df) < 200:
        return []
    
    close = df['close']
    volume = df['Volume']
    signals = []

    # RSI - Slightly loosened for more signals
    rsi = calculate_rsi(close).iloc[-1]
    if rsi > 70:
        signals.append(f"Overbought (RSI {rsi:.1f})")
    elif rsi < 30:
        signals.append(f"Oversold (RSI {rsi:.1f})")

    # MACD Crossover
    macd_line, signal_line = calculate_macd(close)
    if len(macd_line) >= 2:
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals.append("MACD Bullish Crossover")
        elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            signals.append("MACD Bearish Crossover")

    # Golden/Death Cross
    crossover = detect_ma_crossover(df)
    if crossover:
        signals.append(crossover)

    # Volume Spike Confirmation
    if len(volume) >= 20:
        avg_vol = volume.rolling(20).mean().iloc[-1]
        today_vol = volume.iloc[-1]
        if today_vol > avg_vol * 1.5:  # Loosened from 1.8 for more signals
            signals.append(f"Volume Spike ({today_vol / avg_vol:.1f}x avg)")

    if signals:
        current_price = close.iloc[-1]
        return [f"**{ticker}** @ ${current_price:.2f}: " + "; ".join(signals)]
    return []

# ===================== DISCORD BOT =====================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command(name="scan")
async def manual_scan(ctx):
    await ctx.send("🔄 Manual scan started...")
    await daily_analysis()

@bot.command(name="buy")
async def buy_recommendations(ctx):
    await generate_recommendations(ctx, is_buy=True)

@bot.command(name="sell")
async def sell_recommendations(ctx):
    await generate_recommendations(ctx, is_buy=False)

@bot.command(name="news")
async def stock_news(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("Usage: `!news NVDA`")
        return
    ticker = ticker.upper()
    await ctx.send(f"📰 Fetching latest news for **{ticker}**...")
    news = get_stock_news(ticker)
    await ctx.send(news)

async def generate_recommendations(ctx, is_buy: bool):
    channel = ctx.channel
    title = "Buy" if is_buy else "Sell"
    await channel.send(f"🔍 Scanning for strong {title.lower()} signals...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers]
        all_signals = []
        for future in concurrent.futures.as_completed(futures):
            all_signals.extend(future.result())

    if is_buy:
        recommendations = [s for s in all_signals if any(x in s for x in ["Bullish", "Golden Cross", "Oversold", "Volume Spike"])]
    else:
        recommendations = [s for s in all_signals if any(x in s for x in ["Bearish", "Death Cross", "Overbought"])]

    if recommendations:
        msg = f"**Strong {title} Signals**\n" + "\n".join(recommendations[:25])
        if len(recommendations) > 25:
            msg += f"\n... and {len(recommendations)-25} more."
        await channel.send(msg)
    else:
        await channel.send(f"No strong {title.lower()} signals found today.")

async def daily_analysis():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found!")
        return

    await channel.send("📊 **Daily Market Scan Started** 📊")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers]
        all_signals = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_signals.extend(result)

    if all_signals:
        # Prioritize crossovers and strong RSI
        priority = [s for s in all_signals if any(x in s for x in ["Cross", "Overbought", "Oversold"])]
        others = [s for s in all_signals if s not in priority]
        final_list = priority + others
        
        msg = "**High Confidence Signals Today**\n\n" + "\n".join(final_list[:30])
        if len(final_list) > 30:
            msg += f"\n\n... and {len(final_list)-30} more signals."
        await channel.send(msg)
    else:
        await channel.send("✅ No strong technical signals detected today.")

@bot.event
async def on_ready():
    print(f"Bot is online: {bot.user}")
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("🚀 **Stock Analyzer Bot is now LIVE** 🚀\nHigh-confidence signals during market hours.")

async def schedule_daily():
    while True:
        now = datetime.now(timezone.utc)
        # Run during UAE/Dubai trading hours (Sun-Thu, 6-11 UTC)
        if now.weekday() <= 4 and 6 <= now.hour < 11:
            await daily_analysis()
            await asyncio.sleep(3600)  # Wait 1 hour before next scan
        await asyncio.sleep(60)

async def main():
    await asyncio.gather(
        bot.start(DISCORD_TOKEN),
        schedule_daily()
    )

if __name__ == "__main__":
    asyncio.run(main())
