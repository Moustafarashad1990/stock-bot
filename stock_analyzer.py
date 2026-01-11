import discord
from discord.ext import commands, tasks
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import concurrent.futures
import os
from polygon import RESTClient
import yfinance as yf
from textblob import TextBlob
from sklearn.linear_model import LinearRegression  # Simple trend forecast
from peewee import SqliteDatabase, Model, CharField, FloatField, DateTimeField

# ===================== CONFIG =====================
tickers = [
    "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "JPM",
    "AVGO", "V", "WMT", "XOM", "MA", "PG", "JNJ", "HD", "MRK", "ABBV",
    # Add more later - start with 20-100 to avoid overload
    # "A", "ABBV", "ABNB", ... (keep full list commented out)
]

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "1452952904980758571"))
DATA_PERIOD_DAYS = 400
MAX_WORKERS = 10  # Safe for $29 plan
SCAN_TIMEOUT = 300  # 5 minutes total scan timeout

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

# ===================== SIMPLE TREND FORECAST (no fake AI) =====================
def predict_trend_upside(close_prices):
    if len(close_prices) < 30:
        return None
    X = np.arange(len(close_prices)).reshape(-1, 1)
    y = np.array(close_prices)
    model = LinearRegression().fit(X, y)
    future_x = np.array([[len(close_prices) + 30]])  # 30 days ahead
    pred = model.predict(future_x)[0]
    upside = (pred - close_prices[-1]) / close_prices[-1] * 100
    return upside if upside > 20 else None  # Only show if meaningful

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

# ===================== DATA FETCH (Polygon + fallback) =====================
def fetch_polygon_data(ticker: str) -> pd.DataFrame | None:
    try:
        client = RESTClient(POLYGON_API_KEY)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=DATA_PERIOD_DAYS)
        aggs = list(client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            adjusted=True,
            limit=50000
        ))
        if not aggs:
            print(f"Polygon empty for {ticker}")
            return None
        df = pd.DataFrame(aggs)
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df.columns = ['Open', 'High', 'Low', 'close', 'Volume']
        return df
    except Exception as e:
        print(f"Polygon error {ticker}: {e}")
        return None

def fetch_data(ticker: str) -> pd.DataFrame | None:
    df = fetch_polygon_data(ticker)
    if df is not None:
        return df
    # Fallback to yfinance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df
    except:
        print(f"yfinance fallback failed for {ticker}")
        return None

# ===================== SIGNAL PROCESSING =====================
def process_ticker(ticker: str) -> list[str]:
    df = fetch_data(ticker)
    if df is None or len(df) < 200:
        return []
    
    close = df['close']
    volume = df['volume']
    signals = []

    rsi = calculate_rsi(close).iloc[-1]
    if rsi > 70:
        signals.append(f"Overbought (RSI {rsi:.1f})")
    elif rsi < 30:
        signals.append(f"Oversold (RSI {rsi:.1f})")

    macd_line, signal_line = calculate_macd(close)
    if len(macd_line) >= 2:
        if macd_line.iloc[-2] <= signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals.append("MACD Bullish")
        elif macd_line.iloc[-2] >= signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
            signals.append("MACD Bearish")

    crossover = detect_ma_crossover(df)
    if crossover:
        signals.append(crossover)

    if len(volume) >= 20:
        avg_vol = volume.rolling(20).mean().iloc[-1]
        today_vol = volume.iloc[-1]
        if today_vol > avg_vol * 1.5:
            signals.append(f"Volume Spike ({today_vol / avg_vol:.1f}x)")

    upside = predict_trend_upside(close.values)
    if upside:
        signals.append(f"Trend Upside: {upside:.1f}%")

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
    await ctx.send("🔄 Starting full scan... (may take 1-5 minutes)")
    await daily_analysis()

@bot.command(name="buy")
async def buy_recommendations(ctx):
    await ctx.send("🔍 Scanning strong buy signals...")
    await generate_recommendations(ctx, is_buy=True)

@bot.command(name="sell")
async def sell_recommendations(ctx):
    await ctx.send("🔍 Scanning strong sell signals...")
    await generate_recommendations(ctx, is_buy=False)

async def generate_recommendations(ctx, is_buy: bool):
    channel = ctx.channel
    title = "Buy" if is_buy else "Sell"
    keywords = ["Bullish", "Golden Cross", "Oversold"] if is_buy else ["Bearish", "Death Cross", "Overbought"]
    all_signals = []
    try:
        async with asyncio.timeout(300):  # 5 min timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                loop = asyncio.get_event_loop()
                futures = [loop.run_in_executor(executor, process_ticker, t) for t in tickers]
                for future in asyncio.as_completed(futures):
                    result = await future
                    if result:
                        all_signals.extend(result)
    except asyncio.TimeoutError:
        await channel.send("⚠️ Scan timed out - partial results shown")
    recommendations = [s for s in all_signals if any(k in s for k in keywords)]
    if recommendations:
        msg = f"**Strong {title} Signals**\n" + "\n".join(recommendations[:15])
        if len(recommendations) > 15:
            msg += f"\n... and {len(recommendations)-15} more"
        await channel.send(msg)
    else:
        await channel.send(f"No strong {title.lower()} signals found.")

async def daily_analysis():
    channel = bot.get_channel(CHANNEL_ID)
    if not channel:
        print("Channel not found!")
        return
    await channel.send("📊 Daily scan starting...")
    all_signals = []
    try:
        async with asyncio.timeout(300):
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                loop = asyncio.get_event_loop()
                futures = [loop.run_in_executor(executor, process_ticker, t) for t in tickers]
                for future in asyncio.as_completed(futures):
                    result = await future
                    if result:
                        all_signals.extend(result)
    except asyncio.TimeoutError:
        await channel.send("⚠️ Daily scan timed out - partial results")
    if all_signals:
        priority = [s for s in all_signals if any(x in s for x in ["Cross", "Overbought", "Oversold"])]
        others = [s for s in all_signals if s not in priority]
        final_list = priority + others
        msg = "**Today's Signals**\n\n" + "\n".join(final_list[:20])
        if len(final_list) > 20:
            msg += f"\n... {len(final_list)-20} more"
        await channel.send(msg)
    else:
        await channel.send("No strong signals today.")

@tasks.loop(hours=1)
async def scheduled_scan():
    now = datetime.now(timezone.utc)
    if now.weekday() <= 4 and 6 <= now.hour < 11:  # Dubai market hours
        print("Running scheduled scan...")
        channel = bot.get_channel(CHANNEL_ID)
        if channel:
            await channel.send("🕒 Scheduled hourly scan starting...")
            await daily_analysis()

@bot.event
async def on_ready():
    print(f"Bot online: {bot.user}")
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send("🚀 **Stock Bot is LIVE** 🚀\nUse !scan, !buy, !sell")
    scheduled_scan.start()

async def main():
    await bot.start(DISCORD_TOKEN)

if __name__ == "__main__":
    asyncio.run(main())
