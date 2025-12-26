import discord
from discord.ext import commands
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback
import concurrent.futures
import torch
import torch.nn as nn
import finnhub  # pip install finnhub-python

# ===================== CONFIG =====================
tickers = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B",
    "LLY", "WMT", "JPM", "V", "ORCL", "MA", "JNJ", "XOM", "PLTR", "BAC",
    "ABBV", "NFLX", "COST", "AMD", "HD", "PG", "GE", "MU", "CSCO", "KO",
    "CVX", "WFC", "UNH", "MS", "IBM", "CAT", "GS", "AXP", "MRK", "RTX",
    "PM", "CRM", "APP", "MCD", "LRCX", "TMUS", "ABT", "TMO", "C", "ISRG",
    "AMAT", "PEP", "DIS", "LIN", "INTU", "QCOM", "SCHW", "GEV", "AMGN", "BKNG",
    "TJX", "INTC", "T", "BA", "UBER", "BLK", "VZ", "NEE", "ACN", "KLAC",
    "APH", "ANET", "NOW", "TXN", "DHR", "SPGI", "COF", "GILD", "ADBE", "PFE",
    "BSX", "UNP", "SYK", "LOW", "ADI", "PGR", "PANW", "WELL", "DE", "MDT",
    "ETN", "HON", "CB", "CRWD", "BX", "PLD", "KKR", "VRTX", "COP", "NEM" , "AZN" , "HIMS" , "IONQ" , "PYPL" , "NOW" , "SYK" , "ZETA" 
]

import os
DISCORD_TOKEN = str(os.getenv("DISCORD_TOKEN"))
CHANNEL_ID = 1452952904980758571
FINNHUB_API_KEY = "d56imfhr01qu3qoaukjgd56imfhr01qu3qoaukk0"
DATA_PERIOD = "1y"
MAX_WORKERS = 12
SCAN_TIMES = [(19, 0), (22, 0)]  # 7:00 PM and 10:00 PM
MARKET_OPEN_HOUR = 9  # 9:30 AM ET (adjust to your timezone)
MARKET_CLOSE_HOUR = 16  # 4:00 PM ET
REAL_TIME_TICKERS = tickers[:20]  # Limit to top 20 for free tier (expand if premium)

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
lstm_model = FastLSTM().to(device)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
lstm_criterion = nn.MSELoss()

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
        X_tensor = torch.from_numpy(X).unsqueeze(-1)
        y_tensor = torch.from_numpy(y).unsqueeze(-1)
        lstm_model.train()
        lstm_optimizer.zero_grad()
        for _ in range(5):
            output = lstm_model(X_tensor)
            loss = lstm_criterion(output, y_tensor)
            loss.backward()
            lstm_optimizer.step()
            lstm_optimizer.zero_grad()
        lstm_model.eval()
        last_seq = torch.from_numpy(scaled[-lookback:]).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred_scaled = lstm_model(last_seq).item()
        pred_price = pred_scaled * (max_val - min_val) + min_val
        return round(pred_price, 2)
    except:
        return None

# ===================== TICKER PROCESSING =====================
def process_ticker(ticker: str, multi_data) -> list[str]:
    try:
        data = multi_data[ticker].dropna()
        if len(data) < 200:
            return []
        df = pd.DataFrame({
            'open': data['Open'],
            'high': data['High'],
            'low': data['Low'],
            'close': data['Close'],
            'volume': data['Volume']
        }).reset_index(drop=True)

        signals = []

        # RSI
        rsi_val = calculate_rsi(df['close']).iloc[-1]
        if rsi_val > 70:
            signals.append(f"RSI Overbought ({rsi_val:.1f})")
        if rsi_val < 30:
            signals.append(f"RSI Oversold ({rsi_val:.1f})")

        # MACD
        macd, signal = calculate_macd(df['close'])
        if len(macd) >= 2:
            if macd.iloc[-2] <= signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
                signals.append("MACD Bullish Cross")
            if macd.iloc[-2] >= signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
                signals.append("MACD Bearish Cross")

        # MA Crossover
        ma_sig = detect_ma_crossover(df)
        if ma_sig:
            signals.append(ma_sig)

        # LSTM prediction only if other signals exist
        if signals:
            pred = lstm_predict_next_close(df['close'])
            if pred is not None:
                curr = df['close'].iloc[-1]
                change = pred - curr
                direction = "↑" if change > 0 else "↓"
                signals.append(f"LSTM → ${pred} ({direction}{abs(change):.2f})")

        return [f"{ticker}: {s}" for s in signals]
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return []

# ===================== DISCORD BOT =====================
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Manual scan command — type !scan in Discord to run anytime
@bot.command(name="scan")
async def manual_scan(ctx):
    await ctx.send(f"Manual scan triggered by {ctx.author.name}! Running analysis now...")
    await daily_analysis()

# Buy command — type !buy to get buy recommendations
@bot.command(name="buy")
async def buy_recommendations(ctx):
    await ctx.send(f"Fetching buy recommendations for {ctx.author.name}...")
    await generate_recommendations(ctx, is_buy=True)

# Sell command — type !sell to get sell recommendations
@bot.command(name="sell")
async def sell_recommendations(ctx):
    await ctx.send(f"Fetching sell recommendations for {ctx.author.name}...")
    await generate_recommendations(ctx, is_buy=False)

# News command — type !news TICKER
@bot.command(name="news")
async def stock_news(ctx, ticker: str = None):
    if not ticker:
        await ctx.send("Please provide a ticker, e.g., !news AAPL")
        return
    ticker = ticker.upper()
    await ctx.send(f"Fetching latest news for {ticker}...")
    news_text = get_stock_news(ticker)
    await ctx.send(news_text)

async def generate_recommendations(ctx, is_buy: bool):
    channel = ctx.channel
    now = datetime.now()
    title = "Buy" if is_buy else "Sell"
    await channel.send(f"Starting {title} recommendation scan - {now.strftime('%Y-%m-%d %H:%M')}")

    try:
        multi_data = yf.download(" ".join(tickers), period=DATA_PERIOD, group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        await channel.send("Data fetch failed")
        print(traceback.format_exc())
        return

    if datetime.now().weekday() >= 5:
        await channel.send("Weekend - no trading. Skipping analysis.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t, multi_data) for t in tickers]
        all_signals = []
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())

    # Filter for buy/sell recommendations
    recommendations = []
    if is_buy:
        # Buy params: Bullish MACD, Golden Cross, RSI < 70 (not overbought), positive LSTM change
        for sig in all_signals:
            if "MACD Bullish Cross" in sig or "Golden Cross" in sig:
                recommendations.append(sig)
    else:
        # Sell params: Bearish MACD, Death Cross, RSI > 70 (overbought), negative LSTM change
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

    now = datetime.now()
    await channel.send(f"Starting daily scan - {now.strftime('%Y-%m-%d %H:%M')}")

    try:
        multi_data = yf.download(" ".join(tickers), period=DATA_PERIOD, group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        await channel.send("Data fetch failed")
        print(traceback.format_exc())
        return

    if datetime.now().weekday() >= 5:
        await channel.send("Weekend - no trading. Skipping analysis.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_ticker, t, multi_data) for t in tickers]
        all_signals = []
        for f in concurrent.futures.as_completed(futures):
            all_signals.extend(f.result())

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
        await channel.send("Stock Analyzer with LSTM is online! Scans at 7:00 PM & 10:00 PM weekdays ⚡")

async def schedule_daily():
    while True:
        now = datetime.now()
        if now.weekday() < 5:  # Weekdays only
            for hour, minute in SCAN_TIMES:
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                if wait_seconds > 0:
                    print(f"Next scan at {target.strftime('%H:%M on %Y-%m-%d')} in {wait_seconds / 3600:.1f} hours")
                    await asyncio.sleep(wait_seconds)
                    await daily_analysis()
        # After Friday scans, sleep until next Monday's first scan
        await asyncio.sleep(3600)  # Check hourly on weekends

async def main():
    await asyncio.gather(
        bot.start(DISCORD_TOKEN),
        schedule_daily()
    )

if __name__ == "__main__":
    asyncio.run(main())
