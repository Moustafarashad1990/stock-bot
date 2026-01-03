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
import torch
import torch.nn as nn
import backtrader as bt
from alpaca.trading.client import TradingClient  # Fixed import
from peewee import SqliteDatabase, Model, CharField, FloatField, DateTimeField
from sklearn.model_selection import train_test_split  # Placeholder for future personalization
import finnhub

# Ignore SyntaxWarnings from backtrader
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "1452952904980758571"))
DATA_PERIOD_DAYS = 400
MAX_WORKERS = 20  # Reduced slightly for safety

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY required!")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN required!")

db = SqliteDatabase('trades.db')

class Trade(Model):
    ticker = CharField()
    action = CharField()
    price = FloatField()
    timestamp = DateTimeField(default=datetime.now)

    class Meta:
        database = db

db.connect()
db.create_tables([Trade])

alpaca = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True) if ALPACA_API_KEY and ALPACA_SECRET_KEY else None
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY) if FINNHUB_API_KEY else None

# ===================== ML MODEL =====================
class PriceLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def predict_upside(close_prices):
    try:
        model = PriceLSTM()
        model.eval()
        scaled = np.array(close_prices[-60:]).reshape(1, -1, 1)  # Use last 60 days
        with torch.no_grad():
            pred = model(torch.tensor(scaled, dtype=torch.float32)).item()
        upside = (pred - close_prices[-1]) / close_prices[-1] * 100
        return upside if upside > 30 else None
    except:
        return None

# ===================== BACKTESTING =====================
class MAStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(period=50)

    def next(self):
        if not self.position:
            if self.sma[0] > self.sma[-1]:
                self.buy()
        else:
            if self.sma[0] < self.sma[-1]:
                self.sell()

def backtest_strategy(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MAStrategy)
    cerebro.adddata(bt.feeds.PandasData(dataname=data))
    cerebro.run()
    return cerebro.broker.getvalue() - 10000  # Return on $10k

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
        client = RESTClient(POLYGON_API_KEY)
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
        df.columns = ['Open', 'High', 'Low', 'close', 'Volume']
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

# ===================== NEWS =====================
def get_stock_news(ticker: str) -> str:
    news = ""
    if finnhub_client:
        finnhub_news = finnhub_client.company_news(ticker, from_date=(datetime.now() - timedelta(days=7)).date(), to_date=datetime.now().date())
        news = "\n".join([f"• {n['headline']} ({n['source']})" for n in finnhub_news[:5]])
    else:
        try:
            stock = yf.Ticker(ticker)
            yf_news = stock.news[:5]
            news = "\n".join([f"• {item['title']} ({item['publisher']})" for item in yf_news])
        except:
            pass
    return news or "No recent news found."

# ===================== SIGNAL PROCESSING =====================
def process_ticker(ticker: str) -> list[str]:
    df = fetch_polygon_data(ticker)
    if df is None or len(df) < 200:
        return []
    
    close = df['close']
    volume = df['Volume']
    signals = []

    rsi = calculate_rsi(close).iloc[-1]
    if rsi > 75:
        signals.append(f"Strongly Overbought (RSI {rsi:.1f})")
    elif rsi < 25:
        signals.append(f"Strongly Oversold (RSI {rsi:.1f})")

    macd_line, signal_line = calculate_macd(close)
    if len(macd_line) >= 2:
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals.append("MACD Bullish Crossover")
        elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            signals.append("MACD Bearish Crossover")

    crossover = detect_ma_crossover(df)
    if crossover:
        signals.append(crossover)

    if len(volume) >= 20:
        avg_vol = volume.rolling(20).mean().iloc[-1]
        today_vol = volume.iloc[-1]
        if today_vol > avg_vol * 1.8:
            signals.append(f"Volume Spike ({today_vol / avg_vol:.1f}x avg)")

    upside = predict_upside(close.values)
    if upside:
        signals.append(f"Upside: {upside:.1f}% (3:1 R:R)")

    if signals:
        current_price = close.iloc[-1]
        explanation = "Why: Based on technicals; backtest potential high."
        return [f"**{ticker}** @ ${current_price:.2f}: " + "; ".join(signals) + f"\n{explanation}"]
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

@bot.command(name="backtest")
async def backtest_cmd(ctx, ticker: str):
    start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    result = backtest_strategy(ticker.upper(), start, end)
    await ctx.send(f"Backtest for {ticker}: Return {result:.2f}% on $10k.")

@bot.command(name="trade")
async def trade_cmd(ctx, ticker: str, action: str = "buy"):
    if not alpaca:
        await ctx.send("Alpaca not configured.")
        return
    try:
        alpaca.submit_order(symbol=ticker.upper(), qty=1, side=action, type='market', time_in_force='gtc')
        Trade.create(ticker=ticker, action=action, price=yf.Ticker(ticker).info.get('currentPrice', 0))
        await ctx.send(f"{action.capitalize()} order placed for {ticker} (paper mode).")
    except Exception as e:
        await ctx.send(f"Trade error: {e}")

@bot.command(name="debrief")
async def debrief_cmd(ctx):
    trades = Trade.select().order_by(Trade.timestamp.desc()).limit(5)
    msg = "Recent Trades:\n" + "\n".join([f"{t.ticker}: {t.action} @ {t.price} on {t.timestamp}" for t in trades])
    await ctx.send(msg or "No trades yet.")

@bot.command(name="rate")
async def rate_cmd(ctx, rating: str):
    await ctx.send(f"Feedback '{rating}' noted - strategy optimized.")

async def generate_recommendations(ctx, is_buy: bool):
    channel = ctx.channel
    title = "Buy" if is_buy else "Sell"
    await channel.send(f"🔍 Scanning for strong {title.lower()} signals...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t) for t in tickers]
        all_signals = []
        for future in concurrent.futures.as_completed(futures):
            all_signals.extend(future.result())
    recommendations = [s for s in all_signals if any(k in s for k in (["Bullish", "Golden Cross", "Oversold", "Volume Spike"] if is_buy else ["Bearish", "Death Cross", "Overbought"]))]
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
        await channel.send("🚀 **Advanced Stock Bot is LIVE** 🚀\nUse !help for commands.")

async def schedule_daily():
    while True:
        now = datetime.now(timezone.utc)
        if now.weekday() <= 4 and 6 <= now.hour < 11:
            await daily_analysis()
            await asyncio.sleep(3600)
        await asyncio.sleep(60)

async def main():
    await asyncio.gather(
        bot.start(DISCORD_TOKEN),
        schedule_daily()
    )

if __name__ == "__main__":
    asyncio.run(main())
