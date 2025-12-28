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
import yfinance as yf  # For news fallback
import os
from textblob import TextBlob

# ===================== CONFIG =====================
# Full S&P 500 tickers as of December 2025 (expanded from limited list, sourced from Wikipedia and Slickcharts)
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
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "1452952904980758571"))  # Default fallback
DATA_PERIOD_DAYS = 400  # >1 year
MAX_WORKERS = 30  # Increased for paid Polygon (unlimited calls)
# Dubai market hours: Sun-Thu, 06:00-11:00 UTC, scan every hour

if not POLYGON_API_KEY:
    raise ValueError("POLYGON_API_KEY required!")

# ===================== LSTM MODEL =====================
# (LSTM is used in process_ticker to predict next close price if signals are detected)
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
# (Unchanged from original)

# ===================== LSTM PREDICTION =====================
# (Unchanged — predicts next price using LSTM if signals found)

# ===================== POLYGON FETCH =====================
# (Using Polygon for full tickers — paid plan handles 500+ easily)

# ===================== NEWS FUNCTION =====================
# (Unchanged — yfinance for news)

# ===================== TICKER PROCESSING =====================
# (Unchanged — uses LSTM if signals)

# ===================== DISCORD BOT =====================
# (Unchanged — commands, analysis, scheduling with Dubai hours)

# Use the deployment steps from before to upload this (local edit, push, Render deploy).