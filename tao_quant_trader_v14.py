"""
TAO QUANT TRADER — Version 14
TAO market watcher + dashboard/sound/Telegram alerts + V4 analytics and paper trading.

Install in PowerShell:
    py -m pip install --upgrade streamlit yfinance pandas numpy streamlit-autorefresh websocket-client

Run:
    py -m streamlit run tao_quant_trader_v13.py

The app creates/uses tao_paper_trading.db beside this script.
If you already used V3 and keep the V3 database in the same folder,
V14 reuses the same paper portfolio and trade history.

DATA NOTE:
Live market ticks use Binance Spot TAO/USDT public WebSocket and reconnect automatically. Historical indicator candles still use Yahoo Finance and may be delayed. USDT is a USD proxy, not USD.

SAFETY:
Paper trading only. No exchange keys, API trading, or real orders.
"""

import sqlite3
import json
import urllib.parse
import urllib.request
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="TAO Quant Lab V14",
    page_icon="📈",
    layout="wide",
)

TICKER = "TAO22974-USD"
DB_PATH = Path(__file__).with_name("tao_paper_trading.db")
STARTING_CASH = 10_000.0
DAYS_PER_YEAR = 365
IST = ZoneInfo("Asia/Kolkata")
IST_LABEL = "IST (UTC+05:30)"


def now_ist():
    """Current timezone-aware India Standard Time."""
    return datetime.now(IST)


def as_ist_timestamp(value):
    """Convert timestamps to IST. Naive legacy timestamps are treated as UTC."""
    if value is None or value == "":
        return pd.NaT
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST)


def ist_iso(value, timespec=None):
    """Serialize a timestamp in ISO-8601 with the +05:30 offset."""
    ts = as_ist_timestamp(value)
    if pd.isna(ts):
        return None
    return ts.isoformat() if timespec is None else ts.isoformat(timespec=timespec)


def format_ist(value):
    """Human-readable India time for dashboards, records and Telegram."""
    ts = as_ist_timestamp(value)
    if pd.isna(ts):
        return "—"
    return ts.strftime("%Y-%m-%d %H:%M:%S IST")


RECORD_TIME_COLUMNS = {
    "timestamp", "created_at", "sent_at", "entry_time", "entry_bar",
    "last_checked_bar", "exit_time", "event_time", "received_at",
    "candle_start", "updated_at",
}


def records_in_ist(frame):
    """Copy a record table with all timestamp columns rendered in IST."""
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()
    result = frame.copy()
    for col in result.columns:
        if col.lower() in RECORD_TIME_COLUMNS:
            result[col] = result[col].apply(
                lambda value: format_ist(value) if pd.notna(value) and str(value).strip() else "—"
            )
    return result


# ============================== DATA ==============================

@st.cache_data(ttl=60, show_spinner=False)
def load_intraday():
    """Yahoo 5-minute bars, usually covering the latest several days."""
    frame = yf.download(
        TICKER,
        period="5d",
        interval="5m",
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if "Close" not in frame.columns:
        return pd.DataFrame()
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
    frame = frame[cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Close"]).sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(IST)
    return frame


@st.cache_data(ttl=900, show_spinner=False)
def load_daily(period="5y"):
    frame = yf.download(
        TICKER,
        period=period,
        interval="1d",
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
    frame = frame[cols].copy()
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["Close"]).sort_index()
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(IST)
    return frame


def add_daily_indicators(raw, threshold_pct=50.0):
    df = raw.copy()
    df["Raw_Return"] = df["Close"].pct_change()
    df["Flagged"] = df["Raw_Return"].abs() > threshold_pct / 100
    df["Return"] = df["Raw_Return"].mask(df["Flagged"])

    safe_close = df["Close"].copy()
    for pos in np.flatnonzero(df["Flagged"].fillna(False).to_numpy()):
        if pos > 0:
            safe_close.iloc[pos - 1] = np.nan

    for n in [3, 7, 14, 30]:
        df[f"Momentum_{n}D"] = safe_close / safe_close.shift(n) - 1

    for n in [7, 14, 30]:
        df[f"Volatility_{n}D"] = df["Return"].rolling(n).std()

    df["Persistence_30D"] = df["Return"].abs().rolling(30).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = df["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - 100 / (1 + rs)
    return df


def current_candidate_signal(row):
    needed = ["SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_Signal"]
    if any(pd.isna(row.get(c)) for c in needed):
        return "INSUFFICIENT DATA"
    bullish = (
        row["Close"] > row["SMA_50"]
        and row["SMA_20"] > row["SMA_50"]
        and 50 <= row["RSI_14"] < 70
        and row["MACD"] > row["MACD_Signal"]
    )
    bearish = (
        row["Close"] < row["SMA_50"]
        and row["SMA_20"] < row["SMA_50"]
        and row["MACD"] < row["MACD_Signal"]
    )
    if bullish:
        return "BUY CANDIDATE"
    if bearish:
        return "CASH / EXIT CANDIDATE"
    return "HOLD / NO NEW POSITION"


# ============================== PAPER LEDGER ==============================

def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=20)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=20000")
    return con


def init_db():
    with db_connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY CHECK(id=1),
                initial_cash REAL NOT NULL,
                cash REAL NOT NULL,
                units REAL NOT NULL,
                avg_cost REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                side TEXT NOT NULL,
                units REAL NOT NULL,
                fill_price REAL NOT NULL,
                gross_value REAL NOT NULL,
                fee REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                note TEXT NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS signal_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                direction TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                entry_bar TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop REAL NOT NULL,
                target REAL NOT NULL,
                notional REAL NOT NULL,
                quantity REAL NOT NULL,
                status TEXT NOT NULL,
                last_checked_bar TEXT NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                gross_pnl REAL,
                fees REAL,
                net_pnl REAL,
                duration_minutes REAL
            )
        """)
        # Backward-compatible additions for older paper-trade databases.
        existing_cols = {row["name"] for row in con.execute("PRAGMA table_info(signal_trades)").fetchall()}
        for col_name, col_type in (
            ("signal_score", "INTEGER"),
            ("atr_at_entry", "REAL"),
            ("initial_risk_pct", "REAL"),
            ("initial_reward_risk", "REAL"),
        ):
            if col_name not in existing_cols:
                con.execute(f"ALTER TABLE signal_trades ADD COLUMN {col_name} {col_type}")

        con.execute("""
            CREATE TABLE IF NOT EXISTS market_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                direction TEXT NOT NULL,
                score INTEGER NOT NULL,
                price REAL NOT NULL,
                stop REAL NOT NULL,
                target REAL NOT NULL,
                status TEXT NOT NULL,
                signature TEXT NOT NULL UNIQUE
            )
        """)
        alert_cols = {row["name"] for row in con.execute("PRAGMA table_info(market_alerts)").fetchall()}
        for col_name, col_type in (("paper_trade_status", "TEXT"), ("paper_trade_reason", "TEXT")):
            if col_name not in alert_cols:
                con.execute(f"ALTER TABLE market_alerts ADD COLUMN {col_name} {col_type}")
        con.execute("""CREATE TABLE IF NOT EXISTS paper_entry_signatures (
            signature TEXT PRIMARY KEY,
            trade_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS live_market (
            id INTEGER PRIMARY KEY CHECK(id=1), price REAL, bid REAL, ask REAL,
            event_time TEXT, received_at TEXT, source TEXT, status TEXT
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS live_ticks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, price REAL NOT NULL,
            quantity REAL, side TEXT, event_time TEXT NOT NULL,
            received_at TEXT NOT NULL, source TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS live_candles (
            candle_start TEXT PRIMARY KEY, open REAL NOT NULL, high REAL NOT NULL,
            low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
            candle_closed INTEGER NOT NULL, updated_at TEXT NOT NULL
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS telegram_outbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_key TEXT UNIQUE,
            message TEXT NOT NULL, created_at TEXT NOT NULL, sent_at TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT
        )""")
        exists = con.execute("SELECT id FROM portfolio WHERE id=1").fetchone()
        if exists is None:
            now = now_ist().isoformat(timespec="seconds")
            con.execute(
                """INSERT INTO portfolio
                (id, initial_cash, cash, units, avg_cost, realized_pnl, created_at)
                VALUES (1, ?, ?, 0, 0, 0, ?)""",
                (STARTING_CASH, STARTING_CASH, now),
            )

        migrated = con.execute(
            "SELECT value FROM app_state WHERE key='timezone_migration_ist_v1'"
        ).fetchone()
        if not migrated:
            timestamp_columns = {
                "portfolio": ["created_at"],
                "trades": ["timestamp"],
                "signal_trades": ["entry_time", "entry_bar", "last_checked_bar", "exit_time"],
                "market_alerts": ["timestamp"],
                "live_market": ["event_time", "received_at"],
                "live_ticks": ["event_time", "received_at"],
                "live_candles": ["candle_start", "updated_at"],
                "telegram_outbox": ["created_at", "sent_at"],
            }
            for table, columns in timestamp_columns.items():
                for column in columns:
                    rows_to_convert = con.execute(
                        f"SELECT rowid, {column} AS ts FROM {table} WHERE {column} IS NOT NULL"
                    ).fetchall()
                    for item in rows_to_convert:
                        try:
                            converted = ist_iso(item["ts"])
                            if converted:
                                con.execute(
                                    f"UPDATE {table} SET {column}=? WHERE rowid=?",
                                    (converted, item["rowid"])
                                )
                        except Exception:
                            # Preserve malformed values rather than damaging a record.
                            pass

            for state_key in (
                "telegram_last_success", "telegram_last_poll_success", "live_feed_state_time"
            ):
                state_row = con.execute(
                    "SELECT value FROM app_state WHERE key=?", (state_key,)
                ).fetchone()
                if state_row and state_row["value"]:
                    try:
                        con.execute(
                            "UPDATE app_state SET value=? WHERE key=?",
                            (ist_iso(state_row["value"]), state_key)
                        )
                    except Exception:
                        pass

            con.execute(
                "INSERT INTO app_state(key,value) VALUES('timezone_migration_ist_v1','done') "
                "ON CONFLICT(key) DO UPDATE SET value='done'"
            )


def get_portfolio():
    with db_connect() as con:
        return dict(con.execute("SELECT * FROM portfolio WHERE id=1").fetchone())


def get_trade_history():
    with db_connect() as con:
        rows = con.execute("SELECT * FROM trades ORDER BY id DESC").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def paper_buy(usd_amount, reference_price, fee_bps, slippage_bps):
    if usd_amount <= 0 or reference_price <= 0:
        raise ValueError("Enter a positive amount and valid TAO price.")
    fee = usd_amount * fee_bps / 10000
    fill = reference_price * (1 + slippage_bps / 10000)
    qty = usd_amount / fill
    debit = usd_amount + fee

    with db_connect() as con:
        p = con.execute("SELECT * FROM portfolio WHERE id=1").fetchone()
        if debit > p["cash"] + 1e-8:
            raise ValueError(f"Insufficient paper cash. Available: ${p['cash']:,.2f}")
        new_qty = p["units"] + qty
        new_avg = (p["units"] * p["avg_cost"] + usd_amount + fee) / new_qty
        con.execute(
            "UPDATE portfolio SET cash=?, units=?, avg_cost=? WHERE id=1",
            (p["cash"] - debit, new_qty, new_avg),
        )
        con.execute(
            """INSERT INTO trades
            (timestamp, side, units, fill_price, gross_value, fee, realized_pnl, note)
            VALUES (?, 'BUY', ?, ?, ?, ?, 0, ?)""",
            (
                now_ist().isoformat(timespec="seconds"),
                qty, fill, usd_amount, fee,
                "Paper buy; intraday reference price plus assumed slippage",
            ),
        )
    return qty, fill, fee


def paper_sell(qty, reference_price, fee_bps, slippage_bps):
    if qty <= 0 or reference_price <= 0:
        raise ValueError("Enter a positive quantity and valid TAO price.")
    fill = reference_price * (1 - slippage_bps / 10000)
    gross = qty * fill
    fee = gross * fee_bps / 10000

    with db_connect() as con:
        p = con.execute("SELECT * FROM portfolio WHERE id=1").fetchone()
        if qty > p["units"] + 1e-9:
            raise ValueError(f"Insufficient TAO. Available: {p['units']:.8f}")
        realized = qty * (fill - p["avg_cost"]) - fee
        remaining = max(0.0, p["units"] - qty)
        avg = p["avg_cost"] if remaining > 1e-9 else 0.0
        con.execute(
            """UPDATE portfolio
            SET cash=?, units=?, avg_cost=?, realized_pnl=?
            WHERE id=1""",
            (
                p["cash"] + gross - fee,
                remaining,
                avg,
                p["realized_pnl"] + realized,
            ),
        )
        con.execute(
            """INSERT INTO trades
            (timestamp, side, units, fill_price, gross_value, fee, realized_pnl, note)
            VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?)""",
            (
                now_ist().isoformat(timespec="seconds"),
                qty, fill, gross, fee, realized,
                "Paper sell; intraday reference price minus assumed slippage",
            ),
        )
    return gross, fill, fee, realized


def reset_paper_account():
    with db_connect() as con:
        con.execute("DELETE FROM trades")
        now = now_ist().isoformat(timespec="seconds")
        con.execute(
            """UPDATE portfolio
            SET initial_cash=?, cash=?, units=0, avg_cost=0, realized_pnl=0, created_at=?
            WHERE id=1""",
            (STARTING_CASH, STARTING_CASH, now),
        )


# ============================== MARKET WATCHER ==============================

def add_intraday_indicators(frame):
    x = frame.copy()
    x["EMA20"] = x["Close"].ewm(span=20, adjust=False).mean()
    x["EMA50"] = x["Close"].ewm(span=50, adjust=False).mean()
    fast = x["Close"].ewm(span=12, adjust=False).mean()
    slow = x["Close"].ewm(span=26, adjust=False).mean()
    x["MACD"] = fast - slow
    x["MACD_Signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    delta = x["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["RSI"] = 100 - 100 / (1 + rs)
    high, low, close = x["High"], x["Low"], x["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    x["ADX"] = dx.rolling(14).mean()
    x["ATR"] = atr
    if "Volume" in x.columns:
        x["RelVolume"] = x["Volume"] / x["Volume"].rolling(20).mean().replace(0, np.nan)
    else:
        x["RelVolume"] = np.nan
    return x


def score_market_setup(intraday_frame, daily_row, threshold=85):
    """Conservative six-factor trend setup. A high score is not a win probability."""
    if intraday_frame.empty or len(intraday_frame) < 80:
        return {"direction": "WAIT", "score": 0, "reason": "Not enough intraday bars (need about 80)."}
    x = add_intraday_indicators(intraday_frame)
    r = x.iloc[-1]
    required = ["EMA20", "EMA50", "MACD", "MACD_Signal", "RSI", "ADX", "ATR"]
    if any(pd.isna(r.get(k)) for k in required):
        return {"direction": "WAIT", "score": 0, "reason": "Indicators are still warming up or unavailable."}

    daily_up = pd.notna(daily_row.get("SMA_20")) and pd.notna(daily_row.get("SMA_50")) and daily_row["SMA_20"] > daily_row["SMA_50"]
    daily_down = pd.notna(daily_row.get("SMA_20")) and pd.notna(daily_row.get("SMA_50")) and daily_row["SMA_20"] < daily_row["SMA_50"]
    relvol = r.get("RelVolume", np.nan)
    volume_ok = pd.notna(relvol) and relvol >= 1.0
    adx_ok = r["ADX"] >= 20
    # Avoid chasing a candle already stretched far from its short-term mean.
    extension_atr = abs(float(r["Close"])-float(r["EMA20"])) / max(float(r["ATR"]), 1e-12)
    not_chasing = extension_atr <= 1.5

    long_checks = [
        r["Close"] > r["EMA20"] > r["EMA50"],
        r["MACD"] > r["MACD_Signal"],
        52 <= r["RSI"] <= 68,
        adx_ok,
        volume_ok,
        daily_up,
    ]
    short_checks = [
        r["Close"] < r["EMA20"] < r["EMA50"],
        r["MACD"] < r["MACD_Signal"],
        32 <= r["RSI"] <= 48,
        adx_ok,
        volume_ok,
        daily_down,
    ]
    weights = [25, 20, 15, 15, 15, 10]
    long_score = int(sum(w for w, ok in zip(weights, long_checks) if ok))
    short_score = int(sum(w for w, ok in zip(weights, short_checks) if ok))
    direction, score, checks = ("LONG", long_score, long_checks) if long_score > short_score else (("SHORT", short_score, short_checks) if short_score > long_score else ("WAIT", long_score, []))

    # Hard gates: require 5/6 factors, including trend + momentum + daily trend,
    # and reject overextended entries. Slider threshold can make this stricter.
    if direction == "WAIT":
        reason = "Long/short factors are tied; wait for directional agreement."
    elif sum(checks) < 5:
        reason = f"Only {sum(checks)}/6 confirmation factors passed; at least 5/6 are required."
        direction = "WAIT"
    elif not (checks[0] and checks[1] and checks[5]):
        reason = "Required trend, MACD momentum and daily-trend alignment are not all confirmed."
        direction = "WAIT"
    elif not not_chasing:
        reason = f"Price is extended {extension_atr:.2f} ATR from EMA20; avoid chasing."
        direction = "WAIT"
    elif score < threshold:
        reason = f"Confluence score {score}/100 is below the {threshold} threshold."
        direction = "WAIT"
    else:
        reason = "Strong six-factor confluence: trend, MACD, RSI, ADX, relative volume and daily trend; at least 5/6 passed."

    atr = float(r["ATR"])
    price = float(r["Close"])
    stop_distance = max(2.5 * atr, 0.0065 * price)
    # Gross target is 3.5R, leaving some room for fees/slippage while enforcing >=3R net.
    target_distance = 3.5 * stop_distance
    stop = price - stop_distance if direction == "LONG" else price + stop_distance
    target = price + target_distance if direction == "LONG" else price - target_distance
    if direction == "WAIT":
        stop = target = np.nan
    return {
        "direction": direction, "score": score, "long_score": long_score,
        "short_score": short_score, "price": price, "stop": stop, "target": target,
        "rsi": float(r["RSI"]), "adx": float(r["ADX"]),
        "relvol": float(relvol) if pd.notna(relvol) else np.nan,
        "ema20": float(r["EMA20"]), "ema50": float(r["EMA50"]),
        "atr": atr, "extension_atr": extension_atr,
        "macd_bull": bool(r["MACD"] > r["MACD_Signal"]),
        "factor_count": int(sum(checks)) if checks else 0,
        "timestamp": x.index[-1], "reason": reason,
    }


def get_open_signal_trade():
    """Compatibility helper: return the newest open paper position."""
    with db_connect() as con:
        row = con.execute("SELECT * FROM signal_trades WHERE status='OPEN' ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def get_open_signal_trades(limit=20):
    """Return all open paper positions; V14 monitors every position independently."""
    with db_connect() as con:
        rows = con.execute("SELECT * FROM signal_trades WHERE status='OPEN' ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(row) for row in rows]


def get_signal_trade_history(limit=200):
    with db_connect() as con:
        rows = con.execute("SELECT * FROM signal_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def get_stop_cooldown_remaining(minutes=20):
    """Return remaining cooldown after the most recent stop-loss exit."""
    with db_connect() as con:
        row = con.execute(
            "SELECT exit_time FROM signal_trades WHERE status='CLOSED' AND exit_reason='STOP LOSS' "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if not row or not row["exit_time"]:
        return 0.0
    try:
        elapsed = (now_ist() - pd.Timestamp(row["exit_time"]).to_pydatetime()).total_seconds() / 60
        return max(0.0, float(minutes) - elapsed)
    except Exception:
        return 0.0


def estimate_net_reward_risk(entry, stop, target, fee_bps, slippage_bps):
    """Estimate round-trip cost-adjusted R:R using configured per-side fee/slippage assumptions."""
    risk_distance = abs(float(entry) - float(stop)) / float(entry)
    reward_distance = abs(float(target) - float(entry)) / float(entry)
    round_trip_cost = 2.0 * (float(fee_bps) + float(slippage_bps)) / 10000.0
    net_risk = risk_distance + round_trip_cost
    net_reward = reward_distance - round_trip_cost
    if net_risk <= 0 or net_reward <= 0:
        return 0.0
    return net_reward / net_risk


def open_signal_trade(setup, notional, fee_bps, slippage_bps, entry_signature=None):
    """Open only when the stop has breathing room and cost-adjusted R:R is acceptable."""
    if setup.get("direction") not in ("LONG", "SHORT"):
        set_app_state("last_entry_block_reason", "No directional setup.")
        return False
    open_count = len(get_open_signal_trades(100))
    max_open = int(get_app_state("max_concurrent_paper_trades", "5") or 5)
    if open_count >= max_open:
        set_app_state("last_entry_block_reason", f"Maximum concurrent paper trades reached ({max_open}).")
        return False

    if entry_signature:
        with db_connect() as con:
            used = con.execute("SELECT 1 FROM paper_entry_signatures WHERE signature=?", (entry_signature,)).fetchone()
        if used:
            set_app_state("last_entry_block_reason", "This direction already used its paper-trade slot in the current 30-minute signal window.")
            return False

    ref = float(setup["price"])
    slip = float(slippage_bps) / 10000
    entry = ref * (1 + slip if setup["direction"] == "LONG" else 1 - slip)
    stop = float(setup["stop"])
    target = float(setup["target"])
    net_rr = estimate_net_reward_risk(entry, stop, target, fee_bps, slippage_bps)
    if net_rr < 3.0:
        set_app_state("last_entry_block_reason", f"Trade skipped: estimated net reward/risk is {net_rr:.2f}:1 after fees and slippage; minimum is 3.00:1.")
        return False

    bar_ts = ist_iso(setup["timestamp"])
    risk_fraction = abs(entry - stop) / entry
    # Keep planned gross stop-loss risk near 0.5% of the requested paper size.
    # Wider stops therefore automatically mean smaller paper position sizes.
    risk_budget_usd = float(notional) * 0.005
    actual_notional = min(float(notional), risk_budget_usd / max(risk_fraction, 1e-9))
    qty = actual_notional / entry
    risk_pct = risk_fraction * 100
    atr_value = float(setup.get("atr", np.nan))
    with db_connect() as con:
        cur = con.execute("""INSERT INTO signal_trades
            (direction,entry_time,entry_bar,entry_price,stop,target,notional,quantity,status,last_checked_bar,
             signal_score,atr_at_entry,initial_risk_pct,initial_reward_risk)
            VALUES(?,?,?,?,?,?,?,?,'OPEN',?,?,?,?,?)""",
            (setup["direction"], now_ist().isoformat(timespec="seconds"),
             bar_ts, entry, stop, target, actual_notional, qty, bar_ts,
             int(setup.get("score", 0)), atr_value, risk_pct, net_rr))
        if entry_signature:
            con.execute(
                "INSERT OR IGNORE INTO paper_entry_signatures(signature,trade_id,created_at) VALUES(?,?,?)",
                (entry_signature, int(cur.lastrowid), now_ist().isoformat(timespec="seconds")),
            )
    set_app_state("last_entry_block_reason", "")
    return True

def manage_signal_trade(intraday_frame, fee_bps, slippage_bps):
    """Check bars after entry. If SL and TP hit in one bar, conservatively assume SL first."""
    trade = get_open_signal_trade()
    if trade is None or intraday_frame.empty:
        return None
    last_checked = pd.Timestamp(trade["last_checked_bar"])
    bars = intraday_frame.loc[intraday_frame.index > last_checked]
    if bars.empty:
        return None
    exit_info = None
    for bar_time, bar in bars.iterrows():
        high = float(bar.get("High", bar["Close"]))
        low = float(bar.get("Low", bar["Close"]))
        stop_hit = low <= trade["stop"] if trade["direction"] == "LONG" else high >= trade["stop"]
        target_hit = high >= trade["target"] if trade["direction"] == "LONG" else low <= trade["target"]
        if stop_hit or target_hit:
            reason = "STOP LOSS" if stop_hit else "TAKE PROFIT"
            raw_exit = float(trade["stop"] if stop_hit else trade["target"])
            slip = float(slippage_bps) / 10000
            exit_price = raw_exit * (1 - slip if trade["direction"] == "LONG" else 1 + slip)
            qty = float(trade["quantity"])
            entry = float(trade["entry_price"])
            gross = qty * (exit_price - entry) if trade["direction"] == "LONG" else qty * (entry - exit_price)
            entry_fee = float(trade["notional"]) * float(fee_bps) / 10000
            exit_fee = qty * exit_price * float(fee_bps) / 10000
            fees = entry_fee + exit_fee
            net = gross - fees
            start = pd.Timestamp(trade["entry_time"])
            end = pd.Timestamp(bar_time)
            duration = max(0.0, (end - start).total_seconds() / 60)
            with db_connect() as con:
                con.execute("""UPDATE signal_trades SET status='CLOSED', last_checked_bar=?,
                    exit_time=?, exit_price=?, exit_reason=?, gross_pnl=?, fees=?, net_pnl=?, duration_minutes=?
                    WHERE id=? AND status='OPEN'""",
                    (ist_iso(bar_time), now_ist().isoformat(timespec="seconds"),
                     exit_price, reason, gross, fees, net, duration, int(trade["id"])))
            exit_info = {"direction": trade["direction"], "reason": reason, "exit_price": exit_price,
                         "net_pnl": net, "duration_minutes": duration, "id": trade["id"]}
            break
        with db_connect() as con:
            con.execute("UPDATE signal_trades SET last_checked_bar=? WHERE id=? AND status='OPEN'",
                        (ist_iso(bar_time), int(trade["id"])))
    return exit_info


def telegram_api(bot_token, method, params=None, timeout=15):
    """Call Telegram Bot API and return its result with useful error messages."""
    if not bot_token:
        raise ValueError("Enter your Telegram bot token.")
    params = params or {}
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        msg = str(exc)
        if "409" in msg:
            raise RuntimeError("Telegram polling conflict (HTTP 409). Another app may be polling this bot, or a webhook is configured. Stop other bot instances and use the dashboard's webhook repair button if needed.") from exc
        raise RuntimeError(f"Telegram API connection failed: {msg}") from exc
    if not payload.get("ok"):
        description = payload.get("description", "Unknown Telegram API error")
        raise RuntimeError(f"Telegram API error: {description}")
    return payload.get("result")


def send_telegram_message(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        raise ValueError("Enter both Telegram bot token and chat ID first.")
    result = telegram_api(bot_token, "sendMessage", {"chat_id": str(chat_id).strip(), "text": message})
    set_app_state("telegram_last_success", now_ist().isoformat(timespec="seconds"))
    return result


def edit_telegram_message(bot_token, chat_id, message_id, message):
    result = telegram_api(bot_token, "editMessageText", {
        "chat_id": str(chat_id).strip(), "message_id": int(message_id), "text": message
    })
    set_app_state("telegram_last_success", now_ist().isoformat(timespec="seconds"))
    return result


def get_telegram_updates(bot_token, offset=None):
    params = {"timeout": 0, "allowed_updates": json.dumps(["message"])}
    if offset is not None:
        params["offset"] = int(offset)
    return telegram_api(bot_token, "getUpdates", params, timeout=12) or []


def get_app_state(key, default=None):
    with db_connect() as con:
        row = con.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def set_app_state(key, value):
    with db_connect() as con:
        con.execute("INSERT INTO app_state(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def load_telegram_settings():
    """Load remembered Telegram settings from the local SQLite database."""
    raw = get_app_state("telegram_saved_settings", "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def save_telegram_settings(bot_token, chat_id, alerts, commands, preview):
    """Persist Telegram settings so a Streamlit restart/reconnect does not erase them."""
    payload = {
        "bot_token": str(bot_token or "").strip(),
        "chat_id": str(chat_id or "").strip(),
        "alerts": bool(alerts),
        "commands": bool(commands),
        "preview": bool(preview),
    }
    set_app_state("telegram_saved_settings", json.dumps(payload))


def clear_telegram_settings():
    with db_connect() as con:
        con.execute("DELETE FROM app_state WHERE key='telegram_saved_settings'")


def get_telegram_bot_info(bot_token):
    return telegram_api(bot_token, "getMe")


def get_telegram_webhook_info(bot_token):
    return telegram_api(bot_token, "getWebhookInfo")


def remove_telegram_webhook(bot_token):
    return telegram_api(bot_token, "deleteWebhook", {"drop_pending_updates": "false"})


def discover_telegram_chats(bot_token):
    """Read pending updates without advancing the persisted command offset."""
    updates = get_telegram_updates(bot_token, None)
    found = []
    for item in updates:
        msg = item.get("message") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            found.append({"id": str(chat["id"]), "username": chat.get("username", ""), "first_name": chat.get("first_name", ""), "text": msg.get("text", "")})
    return found


def format_telegram_trade_reply(command, reference_price):
    """Build a concise response from the persistent signal-trade ledger."""
    history = get_signal_trade_history(limit=1)
    active = get_open_signal_trade()
    if command in ("status", "/status"):
        if active:
            direction = active["direction"]
            entry = float(active["entry_price"])
            qty = float(active["quantity"])
            mark = float(reference_price)
            gross = qty * (mark-entry) if direction == "LONG" else qty * (entry-mark)
            elapsed = max(0.0, (now_ist()-pd.Timestamp(active["entry_time"])).total_seconds()/60)
            return ("TAO QUANT LAB — PAPER STATUS\n"
                    f"Trade: OPEN {direction} (#{active['id']})\n"
                    f"Entry: ${entry:.4f} | Current mark: ${mark:.4f}\n"
                    f"Stop loss: ${float(active['stop']):.4f}\nTake profit: ${float(active['target']):.4f}\n"
                    f"Estimated gross P/L: ${gross:.2f}\nTime open: {elapsed:.0f} min\n"
                    "Paper simulation only; price feed may be delayed.")
        return "TAO QUANT LAB — PAPER STATUS\nNo trade is currently open.\nUse 'last trade' to see the most recent result."

    if history.empty:
        return "TAO QUANT LAB — LAST TRADE\nNo paper trades have been recorded yet."
    trade = history.iloc[0].to_dict()
    direction = str(trade.get("direction", "—"))
    status = str(trade.get("status", "—"))
    entry = float(trade.get("entry_price") or 0)
    stop = float(trade.get("stop") or 0)
    target = float(trade.get("target") or 0)
    entry_time = str(trade.get("entry_time") or "—")
    if status == "OPEN":
        qty = float(trade.get("quantity") or 0)
        mark = float(reference_price)
        gross = qty * (mark-entry) if direction == "LONG" else qty * (entry-mark)
        elapsed = max(0.0, (now_ist()-pd.Timestamp(entry_time)).total_seconds()/60)
        return (f"TAO QUANT LAB — LAST TRADE #{int(trade['id'])}\n"
                f"Status: OPEN {direction}\nEntry: ${entry:.4f}\nCurrent mark: ${mark:.4f}\n"
                f"Stop loss: ${stop:.4f}\nTake profit: ${target:.4f}\n"
                f"Estimated gross P/L: ${gross:.2f}\nTime open: {elapsed:.0f} min\n"
                "Still running; no SL/TP has been recorded yet. Paper simulation only.")
    exit_price = float(trade.get("exit_price") or 0)
    net = float(trade.get("net_pnl") or 0)
    duration = float(trade.get("duration_minutes") or 0)
    exit_reason = str(trade.get("exit_reason") or "CLOSED")
    return (f"TAO QUANT LAB — LAST TRADE #{int(trade['id'])}\n"
            f"Status: CLOSED {direction}\nResult: {exit_reason}\n"
            f"Entry: ${entry:.4f}\nExit: ${exit_price:.4f}\n"
            f"Stop loss: ${stop:.4f}\nTake profit: ${target:.4f}\n"
            f"Net simulated P/L: ${net:.2f}\nTime taken: {duration:.0f} min\n"
            f"Opened: {format_ist(entry_time)}\nClosed: {format_ist(trade.get('exit_time'))}\n"
            "Paper simulation only; not a real exchange trade.")


def handle_telegram_commands(bot_token, allowed_chat_id, reference_price):
    """Poll Telegram and answer commands; persist offset across Streamlit reruns/restarts."""
    if not bot_token or not allowed_chat_id:
        return 0
    saved_offset = get_app_state("telegram_update_offset")
    offset = int(saved_offset) if saved_offset is not None else None
    updates = get_telegram_updates(bot_token, offset)
    set_app_state("telegram_last_success", now_ist().isoformat(timespec="seconds"))
    handled = 0
    for update in updates:
        update_id = int(update.get("update_id", 0))
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        incoming_chat_id = str(chat.get("id", ""))
        text_in = str(message.get("text", "")).strip().lower()
        # Advance for every update, including messages from other chats, to avoid polling it forever.
        set_app_state("telegram_update_offset", update_id + 1)
        if incoming_chat_id != str(allowed_chat_id).strip() or not text_in:
            continue
        command = text_in.split("@", 1)[0].strip()
        if command in ("last trade", "last", "/last", "/lasttrade", "/last_trade"):
            reply = format_telegram_trade_reply("last", reference_price)
        elif command in ("status", "/status", "is it running", "current trade", "live", "/live", "open trade"):
            reply = format_telegram_trade_reply("status", reference_price)
        elif command in ("help", "/help", "commands", "/start"):
            reply = ("TAO QUANT LAB — COMMANDS\n"
                     "last trade or /last — latest trade details\n"
                     "status or /status — current open trade and live P/L estimate\n"
                     "live or /live — same as status\n"
                     "help or /help — this help\n\n"
                     "The Streamlit dashboard must be running and Telegram polling enabled. Paper simulation only.")
        else:
            reply = "Command not recognised. Try: last trade, status, live, or help."
        send_telegram_message(bot_token, incoming_chat_id, reply)
        handled += 1
    return handled


def build_live_trade_preview(trade, reference_price, price_timestamp, price_source, fee_bps):
    direction = trade["direction"]
    entry = float(trade["entry_price"])
    qty = float(trade["quantity"])
    mark = float(reference_price)
    gross = qty * (mark-entry) if direction == "LONG" else qty * (entry-mark)
    entry_fee = float(trade["notional"]) * float(fee_bps) / 10000
    exit_fee = qty * mark * float(fee_bps) / 10000
    net = gross - entry_fee - exit_fee
    elapsed = max(0.0, (now_ist()-pd.Timestamp(trade["entry_time"])).total_seconds()/60)
    stop = float(trade["stop"])
    target = float(trade["target"])
    return (f"📡 TAO QUANT LAB — LIVE PAPER TRADE\n"
            f"Trade #{int(trade['id'])} · OPEN {direction}\n"
            f"Entry: ${entry:.4f}\nCurrent: ${mark:.4f}\n"
            f"Stop loss: ${stop:.4f}\nTake profit: ${target:.4f}\n"
            f"Estimated net P/L: ${net:+.2f}\nTime open: {elapsed:.0f} min\n"
            f"Latest bar: {format_ist(price_timestamp)}\nFeed: {price_source}\n"
            "Auto-updated while the dashboard is running. Paper trade only; feed may be delayed.")


def update_telegram_live_preview(bot_token, chat_id, trade, reference_price, price_timestamp, price_source, fee_bps):
    """Create one Telegram message, then edit it on refresh instead of spamming the chat."""
    if not (bot_token and chat_id and trade):
        return False
    text_out = build_live_trade_preview(trade, reference_price, price_timestamp, price_source, fee_bps)
    stored_chat = get_app_state("telegram_preview_chat_id")
    stored_mid = get_app_state("telegram_preview_message_id")
    if stored_chat == str(chat_id).strip() and stored_mid:
        try:
            edit_telegram_message(bot_token, chat_id, stored_mid, text_out)
            return True
        except Exception as exc:
            # If Telegram says the old message cannot be edited, create a fresh preview.
            if "message is not modified" in str(exc).lower():
                return True
    sent = send_telegram_message(bot_token, chat_id, text_out)
    set_app_state("telegram_preview_chat_id", str(chat_id).strip())
    set_app_state("telegram_preview_message_id", sent.get("message_id", ""))
    return True


def clear_telegram_live_preview():
    set_app_state("telegram_preview_chat_id", "")
    set_app_state("telegram_preview_message_id", "")


def record_alert_once(setup):
    if setup.get("direction") not in ("LONG", "SHORT"):
        return False
    # One alert per direction and closed 1-minute signal candle.
    signature = market_signal_signature(setup)
    with db_connect() as con:
        try:
            con.execute(
                "INSERT INTO market_alerts(timestamp,direction,score,price,stop,target,status,signature) VALUES(?,?,?,?,?,?,?,?)",
                (now_ist().isoformat(timespec="seconds"), setup["direction"], int(setup["score"]),
                 float(setup["price"]), float(setup["stop"]), float(setup["target"]), "NEW", signature),
            )
            return True
        except sqlite3.IntegrityError:
            return False


def get_market_alerts(limit=50):
    with db_connect() as con:
        rows = con.execute("SELECT * FROM market_alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def market_signal_signature(setup):
    """One simulated entry per direction and closed 1-minute signal candle."""
    bar = ist_iso(pd.Timestamp(setup["timestamp"]).floor("min"))
    return f"{setup['direction']}|{bar}"


def update_alert_paper_status(signature, status, reason=""):
    with db_connect() as con:
        con.execute(
            "UPDATE market_alerts SET paper_trade_status=?, paper_trade_reason=? WHERE signature=?",
            (status, str(reason)[:500], signature),
        )


# ============================== CONTINUOUS LIVE SERVICES ==============================


def set_live_state(status, detail=""):
    set_app_state("live_feed_status", status)
    set_app_state("live_feed_detail", detail[:400])
    set_app_state("live_feed_state_time", now_ist().isoformat(timespec="seconds"))


def store_live_market(price=None, bid=None, ask=None, event_time=None, quantity=None, side=None, source="Binance Spot WebSocket"):
    now = now_ist().isoformat(timespec="milliseconds")
    with db_connect() as con:
        old = con.execute("SELECT * FROM live_market WHERE id=1").fetchone()
        last_price = float(price) if price is not None else (float(old["price"]) if old and old["price"] is not None else None)
        last_bid = float(bid) if bid is not None else (float(old["bid"]) if old and old["bid"] is not None else None)
        last_ask = float(ask) if ask is not None else (float(old["ask"]) if old and old["ask"] is not None else None)
        et = event_time or (old["event_time"] if old else now)
        con.execute("""INSERT INTO live_market(id,price,bid,ask,event_time,received_at,source,status)
            VALUES(1,?,?,?,?,?,?, 'CONNECTED')
            ON CONFLICT(id) DO UPDATE SET price=excluded.price,bid=excluded.bid,ask=excluded.ask,
            event_time=excluded.event_time,received_at=excluded.received_at,source=excluded.source,status='CONNECTED'""",
            (last_price,last_bid,last_ask,et,now,source))
        if price is not None:
            con.execute("INSERT INTO live_ticks(price,quantity,side,event_time,received_at,source) VALUES(?,?,?,?,?,?)",
                        (float(price), quantity, side, et, now, source))
            con.execute("DELETE FROM live_ticks WHERE id NOT IN (SELECT id FROM live_ticks ORDER BY id DESC LIMIT 3000)")
    if price is not None:
        monitor_open_trade_on_tick(float(price), et)


def get_live_quote():
    with db_connect() as con:
        row = con.execute("SELECT * FROM live_market WHERE id=1").fetchone()
    if not row:
        return None
    q = dict(row)
    try:
        age = max(0.0, (now_ist()-pd.Timestamp(q["event_time"])).total_seconds())
    except Exception:
        age = 999999.0
    try:
        q["connection_age_seconds"] = max(0.0, (now_ist()-pd.Timestamp(q["received_at"])).total_seconds())
    except Exception:
        q["connection_age_seconds"] = 999999.0
    q["age_seconds"] = age
    # Binance sends bookTicker updates independently of trades. The last trade timestamp
    # can be old in a quiet market even while bid/ask is streaming normally. Use the
    # receive age as the connection-health check; keep trade age as diagnostics only.
    q["fresh"] = q.get("price") is not None and q["connection_age_seconds"] <= 20
    q["book_fresh"] = q["connection_age_seconds"] <= 20 and q.get("bid") is not None and q.get("ask") is not None
    return q


def get_live_ticks(limit=1000):
    with db_connect() as con:
        rows = con.execute("SELECT * FROM (SELECT * FROM live_ticks ORDER BY id DESC LIMIT ?) ORDER BY id", (int(limit),)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def store_live_candle(start_ms, open_price, high, low, close, volume, is_closed):
    start = datetime.fromtimestamp(float(start_ms)/1000, tz=IST).isoformat(timespec="seconds")
    now = now_ist().isoformat(timespec="seconds")
    with db_connect() as con:
        con.execute("""INSERT INTO live_candles(candle_start,open,high,low,close,volume,candle_closed,updated_at)
            VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(candle_start) DO UPDATE SET
            open=excluded.open,high=excluded.high,low=excluded.low,close=excluded.close,
            volume=excluded.volume,candle_closed=excluded.candle_closed,updated_at=excluded.updated_at""",
            (start,float(open_price),float(high),float(low),float(close),float(volume),int(bool(is_closed)),now))
        con.execute("DELETE FROM live_candles WHERE candle_start NOT IN (SELECT candle_start FROM live_candles ORDER BY candle_start DESC LIMIT 1200)")


def get_live_candles(limit=500, closed_only=True):
    query = "SELECT * FROM live_candles"
    if closed_only:
        query += " WHERE candle_closed=1"
    query += " ORDER BY candle_start DESC LIMIT ?"
    with db_connect() as con:
        rows = con.execute(query,(int(limit),)).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(r) for r in rows]).sort_values("candle_start")
    frame.index = pd.to_datetime(frame.pop("candle_start"),utc=True)
    frame = frame.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    return frame[["Open","High","Low","Close","Volume"]].astype(float)


def bootstrap_binance_candles():
    """Seed the 1-minute indicator history from Binance's public klines REST endpoint."""
    url = "https://api.binance.com/api/v3/klines?symbol=TAOUSDT&interval=1m&limit=500"
    req = urllib.request.Request(url, headers={"User-Agent":"TAO-Quant-Lab/11.0"})
    with urllib.request.urlopen(req, timeout=12) as response:
        rows = json.loads(response.read().decode("utf-8"))
    now_ms = time.time()*1000
    for row in rows:
        store_live_candle(row[0],row[1],row[2],row[3],row[4],row[5],float(row[6]) < now_ms)


def queue_telegram_message(event_key, message):
    with db_connect() as con:
        con.execute("INSERT OR IGNORE INTO telegram_outbox(event_key,message,created_at) VALUES(?,?,?)",
                    (str(event_key), str(message), now_ist().isoformat(timespec="seconds")))


def _close_paper_trade(trade, raw_exit, reason, event_time):
    """Close one paper position exactly once and persist the exchange event time."""
    direction = trade["direction"]
    slip_bps = float(get_app_state("paper_slippage_bps", "15") or 15)
    slip = slip_bps / 10000
    exit_price = float(raw_exit) * (1 - slip if direction == "LONG" else 1 + slip)
    qty, entry = float(trade["quantity"]), float(trade["entry_price"])
    gross = qty * (exit_price-entry) if direction == "LONG" else qty * (entry-exit_price)
    fee_bps = float(get_app_state("paper_fee_bps", "10") or 10)
    fees = float(trade["notional"])*fee_bps/10000 + qty*exit_price*fee_bps/10000
    net = gross-fees
    start = pd.Timestamp(trade["entry_time"])
    end = pd.Timestamp(event_time)
    duration = max(0.0, (end-start).total_seconds()/60)
    event_iso = ist_iso(event_time)
    with db_connect() as con:
        cur = con.execute("""UPDATE signal_trades SET status='CLOSED',last_checked_bar=?,exit_time=?,exit_price=?,
            exit_reason=?,gross_pnl=?,fees=?,net_pnl=?,duration_minutes=? WHERE id=? AND status='OPEN'""",
            (event_iso, event_iso, exit_price, reason, gross, fees, net, duration, int(trade["id"])))
    if cur.rowcount:
        queue_telegram_message(f"trade-close-{trade['id']}",
            f"🏁 TAO PAPER TRADE CLOSED\nTrade #{trade['id']} · {direction}\nResult: {reason}\nEntry: ${entry:.4f}\nExit: ${exit_price:.4f}\nNet simulated P/L: ${net:+.2f}\nDuration: {duration:.1f} minutes\nExit time: {format_ist(event_time)}\nPaper simulation only.")
        return {"id": trade["id"], "reason": reason, "net_pnl": net, "duration_minutes": duration}
    return None


def monitor_open_trade_on_tick(price, event_time):
    """Monitor ALL open positions on every exchange trade tick."""
    closed = []
    for trade in get_open_signal_trades(100):
        direction = trade["direction"]
        stop, target = float(trade["stop"]), float(trade["target"])
        stop_hit = price <= stop if direction == "LONG" else price >= stop
        target_hit = price >= target if direction == "LONG" else price <= target
        if stop_hit or target_hit:
            reason = "STOP LOSS" if stop_hit else "TAKE PROFIT"
            # Stop takes priority if the observed tick has crossed both levels.
            raw_exit = float(price)
            if target_hit and not stop_hit:
                raw_exit = target
            result = _close_paper_trade(trade, raw_exit, reason, event_time)
            if result:
                closed.append(result)
    return closed or None


def monitor_open_trades_on_candle(high, low, close_time):
    """Candle safety net: detect SL/TP touches if trade-tick messages were missed.
    If both levels occur inside one 1-minute candle, conservatively record SL first.
    """
    closed = []
    for trade in get_open_signal_trades(100):
        direction = trade["direction"]
        stop, target = float(trade["stop"]), float(trade["target"])
        stop_hit = low <= stop if direction == "LONG" else high >= stop
        target_hit = high >= target if direction == "LONG" else low <= target
        if stop_hit or target_hit:
            reason = "STOP LOSS" if stop_hit else "TAKE PROFIT"
            raw_exit = stop if stop_hit else target
            result = _close_paper_trade(trade, raw_exit, reason, close_time)
            if result:
                closed.append(result)
    return closed or None


class BinanceLiveFeed:
    """Persistent public Binance TAO/USDT trade + best-bid/ask WebSocket feed."""
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self.run, name="tao-live-websocket", daemon=True)
        self.thread.start()

    def run(self):
        backoff = 1
        while not self.stop_event.is_set():
            try:
                import websocket
                url = "wss://stream.binance.com:9443/stream?streams=taousdt@trade/taousdt@bookTicker/taousdt@kline_1m"
                set_live_state("CONNECTING", "Opening Binance TAO/USDT public WebSocket")
                try:
                    bootstrap_binance_candles()
                except Exception as seed_error:
                    set_app_state("live_candle_bootstrap_error", str(seed_error)[:300])
                ws = websocket.create_connection(url, timeout=12, enable_multithread=True)
                ws.settimeout(20)
                set_live_state("CONNECTED", "Binance TAO/USDT WebSocket connected")
                backoff = 1
                while not self.stop_event.is_set():
                    raw = ws.recv()
                    if not raw:
                        continue
                    packet = json.loads(raw)
                    data = packet.get("data", packet)
                    event = data.get("e", "")
                    if event == "trade":
                        ts = datetime.fromtimestamp(float(data.get("T", data.get("E", time.time()*1000)))/1000, tz=IST).isoformat(timespec="milliseconds")
                        store_live_market(price=float(data["p"]), quantity=float(data.get("q",0)),
                                          side="SELL aggressor" if data.get("m") else "BUY aggressor",
                                          event_time=ts)
                    elif event == "bookTicker":
                        store_live_market(bid=float(data["b"]), ask=float(data["a"]))
                    elif event == "kline":
                        k = data.get("k", {})
                        if k:
                            store_live_candle(k["t"],k["o"],k["h"],k["l"],k["c"],k["v"],k["x"])
                            # The 1m candle stream is a backup monitor if trade events were missed.
                            # Use candle close event time; SL wins if both SL and TP touched.
                            if bool(k.get("x")):
                                candle_event_time = datetime.fromtimestamp(float(k.get("T", time.time()*1000))/1000, tz=IST).isoformat(timespec="milliseconds")
                                monitor_open_trades_on_candle(float(k["h"]), float(k["l"]), candle_event_time)
                try: ws.close()
                except Exception: pass
            except Exception as exc:
                set_live_state("RECONNECTING", f"{type(exc).__name__}: {exc}; retry in {backoff}s")
                self.stop_event.wait(backoff)
                backoff = min(backoff*2, 30)


class TelegramBackgroundService:
    """One long-polling worker with durable offset, retry/backoff and live-preview edits."""
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.thread = threading.Thread(target=self.run, name="tao-telegram-worker", daemon=True)
        self.thread.start()

    def process_outbox(self, token, chat):
        with db_connect() as con:
            rows = con.execute("SELECT * FROM telegram_outbox WHERE sent_at IS NULL ORDER BY id LIMIT 10").fetchall()
        for row in rows:
            try:
                send_telegram_message(token, chat, row["message"])
                with db_connect() as con:
                    con.execute("UPDATE telegram_outbox SET sent_at=?,last_error=NULL WHERE id=?",
                                (now_ist().isoformat(timespec="seconds"),row["id"]))
            except Exception as exc:
                with db_connect() as con:
                    con.execute("UPDATE telegram_outbox SET attempts=attempts+1,last_error=? WHERE id=?",(str(exc)[:400],row["id"]))
                break

    def send_preview(self, token, chat):
        settings = load_telegram_settings()
        if not settings.get("preview"):
            return
        trade = get_open_signal_trade()
        if not trade:
            return
        quote = get_live_quote()
        if not quote or not quote.get("fresh"):
            return
        mark = float(quote["price"])
        direction = trade["direction"]
        entry, qty = float(trade["entry_price"]), float(trade["quantity"])
        gross = qty*(mark-entry) if direction == "LONG" else qty*(entry-mark)
        fee_bps = float(get_app_state("paper_fee_bps", "10") or 10)
        net = gross - float(trade["notional"])*fee_bps/10000 - qty*mark*fee_bps/10000
        elapsed = max(0.0,(now_ist()-pd.Timestamp(trade["entry_time"])).total_seconds()/60)
        text_out = (f"📡 TAO QUANT LAB — LIVE PAPER TRADE\nTrade #{trade['id']} · OPEN {direction}\n"
                    f"TAO/USDT: ${mark:.6f}\nEntry: ${entry:.6f}\nBid/Ask: ${float(quote['bid'] or 0):.6f} / ${float(quote['ask'] or 0):.6f}\n"
                    f"Stop: ${float(trade['stop']):.6f}\nTarget: ${float(trade['target']):.6f}\n"
                    f"Estimated net P/L: ${net:+.2f}\nTime open: {elapsed:.1f} min\n"
                    f"Price age: {quote['age_seconds']:.1f}s · Feed: Binance Spot WebSocket\n"
                    "Paper trade only; not an exchange position.")
        last_edit = float(get_app_state("telegram_preview_last_edit", "0") or 0)
        if time.time()-last_edit < 10: return
        mid = get_app_state("telegram_preview_message_id", "")
        stored_chat = get_app_state("telegram_preview_chat_id", "")
        try:
            if mid and stored_chat == str(chat):
                edit_telegram_message(token, chat, mid, text_out)
            else:
                sent = send_telegram_message(token, chat, text_out)
                set_app_state("telegram_preview_chat_id", str(chat))
                set_app_state("telegram_preview_message_id", str(sent.get("message_id", "")))
            set_app_state("telegram_preview_last_edit", str(time.time()))
        except Exception as exc:
            if "message is not modified" in str(exc).lower():
                set_app_state("telegram_preview_last_edit", str(time.time()))
            else:
                set_app_state("telegram_worker_error", str(exc)[:400])

    def answer_commands(self, token, chat):
        offset = get_app_state("telegram_update_offset")
        params = {"timeout": 3, "allowed_updates": json.dumps(["message"])}
        if offset is not None: params["offset"] = int(offset)
        updates = telegram_api(token, "getUpdates", params, timeout=7) or []
        now_utc = now_ist().isoformat(timespec="seconds")
        set_app_state("telegram_last_poll_success", now_utc)
        for upd in updates:
            uid = int(upd.get("update_id",0))
            msg = upd.get("message") or {}
            ch = msg.get("chat") or {}
            incoming = str(ch.get("id", ""))
            txt = str(msg.get("text", "")).strip().lower().split("@",1)[0]
            if incoming != str(chat).strip() or not txt:
                set_app_state("telegram_update_offset", uid+1)
                continue
            quote = get_live_quote()
            price = float(quote["price"]) if quote and quote.get("price") is not None else np.nan
            if txt in ("last trade","last","/last","/lasttrade","/last_trade"):
                reply = format_telegram_trade_reply("last", price)
            elif txt in ("status","/status","is it running","current trade","live","/live","open trade"):
                reply = format_telegram_trade_reply("status", price)
                if quote: reply += f"\nLive feed age: {quote['age_seconds']:.1f}s · Binance TAO/USDT"
            elif txt in ("help","/help","commands","/start"):
                reply = ("TAO QUANT LAB COMMANDS\n/last — last trade\n/status — active trade status\n/live — live trade preview\n/help — commands\n\nCommands are handled by the background Telegram service while this app process is running.")
            else:
                reply = "Unknown command. Try /last, /status, /live or /help."
            send_telegram_message(token, incoming, reply)
            # Advance only after successful reply delivery so a transient send failure is retried.
            set_app_state("telegram_update_offset", uid+1)

    def run(self):
        backoff = 2
        while not self.stop_event.is_set():
            # Connection intent is persisted separately from credentials. Temporary
            # network failures never clear this flag; only the user's Disconnect button does.
            connection_requested = get_app_state("telegram_connection_enabled", "0") == "1"
            if not connection_requested:
                set_app_state("telegram_worker_status", "DISCONNECTED BY USER")
                self.stop_event.wait(2)
                continue

            settings = load_telegram_settings()
            token, chat = settings.get("bot_token", ""), settings.get("chat_id", "")
            if not token or not chat:
                set_app_state("telegram_worker_status", "WAITING FOR SAVED TOKEN / CHAT ID")
                self.stop_event.wait(2)
                continue
            try:
                # Drain durable notifications even when command replies are disabled.
                if settings.get("alerts") or settings.get("preview"):
                    self.process_outbox(token, chat)
                if settings.get("preview"):
                    self.send_preview(token, chat)
                if settings.get("commands", True):
                    self.answer_commands(token, chat)
                    set_app_state("telegram_worker_status", "CONNECTED / COMMANDS ON")
                else:
                    set_app_state("telegram_worker_status", "CONNECTED / NOTIFICATIONS ONLY")
                    self.stop_event.wait(2)
                set_app_state("telegram_worker_error", "")
                backoff = 2
            except Exception as exc:
                set_app_state("telegram_worker_status", "RETRYING")
                set_app_state("telegram_worker_error", str(exc)[:400])
                self.stop_event.wait(backoff)
                backoff = min(backoff*2, 30)


class BackgroundServices:
    def __init__(self):
        self.feed = BinanceLiveFeed()
        self.telegram = TelegramBackgroundService()
        self.feed.start()
        self.telegram.start()


@st.cache_resource(show_spinner=False)
def get_background_services():
    return BackgroundServices()


# ============================== ANALYTICS ==============================

def backtest_long_cash(df, fee_bps, slippage_bps):
    """Simple close-generated rule applied to next day's return."""
    x = df.copy()
    x["Signal"] = x.apply(current_candidate_signal, axis=1)
    x["Target_Position"] = (x["Signal"] == "BUY CANDIDATE").astype(float)
    x["Position"] = x["Target_Position"].shift(1).fillna(0.0)
    x["Turnover"] = x["Position"].diff().abs()
    if len(x):
        x.iloc[0, x.columns.get_loc("Turnover")] = abs(x["Position"].iloc[0])
    cost = (fee_bps + slippage_bps) / 10000
    x["Strategy_Return"] = (
        x["Position"] * x["Return"] - x["Turnover"] * cost
    )
    x["BuyHold_Return"] = x["Return"]
    x = x.dropna(subset=["Strategy_Return", "BuyHold_Return"]).copy()
    x["Strategy_Equity"] = (1 + x["Strategy_Return"]).cumprod()
    x["BuyHold_Equity"] = (1 + x["BuyHold_Return"]).cumprod()
    return x


def stats(returns, equity):
    returns = returns.dropna()
    if len(returns) == 0:
        return {}
    total = equity.iloc[-1] - 1
    years = len(returns) / DAYS_PER_YEAR
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    dd = equity / equity.cummax() - 1
    vol = returns.std()
    sharpe = returns.mean() / vol * np.sqrt(DAYS_PER_YEAR) if pd.notna(vol) and vol > 0 else np.nan
    return {
        "Total return": total,
        "Annualized return": cagr,
        "Max drawdown": dd.min(),
        "Sharpe (rough)": sharpe,
    }


def money(v):
    return "—" if pd.isna(v) else f"${v:,.2f}"


def percent(v):
    return "—" if pd.isna(v) else f"{v * 100:.2f}%"


# ============================== UI ==============================

init_db()
services = get_background_services()

st.title("📈 TAO QUANT TRADER")
st.caption("Version 14 · India Standard Time (IST) throughout records, dashboard and exports · strict net 3:1 minimum R:R · bookTicker-aware feed freshness · 5/6 factor confirmation · anti-chase filter · multi-position auto-paper + tick/candle SL/TP monitor + event-time audit log · Telegram preserved · Binance WebSocket TAO/USDT")

with st.sidebar:
    st.header("Market settings")
    history = st.selectbox("Daily analytics history", ["2y", "5y", "max"], index=1)
    flag_threshold = st.number_input(
        "Flag daily returns above (%)", 10.0, 300.0, 50.0, 5.0,
        help="Flagged observations are excluded from cleaned return and backtest calculations.",
    )
    st.header("Market watcher")
    alert_threshold = st.slider("Minimum setup score", 85, 100, 85, 5)
    enable_sound = st.checkbox("Enable browser sound alerts", value=True)
    auto_refresh = st.checkbox("Auto-refresh dashboard every 1 second", value=True)
    st.header("Automatic signal paper-trades")
    auto_paper = st.checkbox("Automatically paper-trade every qualifying fresh LONG/SHORT signal", value=True)
    paper_notional = st.number_input("Paper trade size (USDT)", 25.0, 10000.0, 250.0, 25.0)
    max_open_positions = st.number_input("Maximum simultaneous paper trades", 1, 20, 5, 1)
    set_app_state("max_concurrent_paper_trades", str(int(max_open_positions)))
    st.caption("Paper only. Entry requires a fresh Binance quote, at least 5/6 setup factors, and estimated NET reward/risk of at least 3:1 after configured fees/slippage. No trade is forced.")
    st.header("Telegram connection")
    # Restore saved credentials and connection intent across Streamlit reruns/restarts.
    remembered = load_telegram_settings()
    if "telegram_remember" not in st.session_state:
        st.session_state.telegram_remember = True
    if "telegram_token_input" not in st.session_state:
        st.session_state.telegram_token_input = remembered.get("bot_token", "")
    if "telegram_chat_input" not in st.session_state:
        st.session_state.telegram_chat_input = remembered.get("chat_id", "")
    if "telegram_alerts_input" not in st.session_state:
        st.session_state.telegram_alerts_input = bool(remembered.get("alerts", True))
    if "telegram_commands_input" not in st.session_state:
        st.session_state.telegram_commands_input = bool(remembered.get("commands", True))
    if "telegram_preview_input" not in st.session_state:
        st.session_state.telegram_preview_input = bool(remembered.get("preview", True))

    # Migration: existing V9 installations with saved credentials are automatically
    # marked connected once, so users do not have to configure the bot again.
    saved_connection_state = get_app_state("telegram_connection_enabled")
    if saved_connection_state is None and remembered.get("bot_token") and remembered.get("chat_id"):
        set_app_state("telegram_connection_enabled", "1")
    elif saved_connection_state is None:
        set_app_state("telegram_connection_enabled", "0")

    telegram_remember = st.checkbox(
        "Remember Telegram on this computer",
        key="telegram_remember",
        help="Required for automatic reconnection after Streamlit restarts. Credentials are stored in your local SQLite database.",
    )
    telegram_token = st.text_input(
        "Telegram bot token", type="password", key="telegram_token_input",
        help="Keep this token private. It is saved locally only when Remember is enabled.",
    )
    telegram_chat = st.text_input(
        "Telegram chat ID", key="telegram_chat_input",
        help="Your numeric Telegram chat ID.",
    )
    telegram_enabled = st.checkbox("Send LONG/SHORT alerts", key="telegram_alerts_input")
    telegram_commands_enabled = st.checkbox("Enable Telegram commands", key="telegram_commands_input")
    telegram_live_preview = st.checkbox("Update live trade preview message", key="telegram_preview_input")

    connection_requested = get_app_state("telegram_connection_enabled", "0") == "1"
    if telegram_remember:
        save_telegram_settings(
            telegram_token, telegram_chat, telegram_enabled,
            telegram_commands_enabled, telegram_live_preview
        )
        st.caption("Credentials and notification preferences are saved locally.")
    else:
        if connection_requested:
            set_app_state("telegram_connection_enabled", "0")
        clear_telegram_settings()
        connection_requested = False
        st.caption("Turn on Remember to keep Telegram connected after an app restart.")

    connect_col, disconnect_col = st.columns(2)
    with connect_col:
        if st.button(
            "Connect / Reconnect",
            use_container_width=True,
            disabled=(not telegram_remember or not telegram_token.strip() or not telegram_chat.strip()),
        ):
            try:
                bot = get_telegram_bot_info(telegram_token.strip())
                webhook = get_telegram_webhook_info(telegram_token.strip())
                if webhook and webhook.get("url"):
                    remove_telegram_webhook(telegram_token.strip())
                save_telegram_settings(
                    telegram_token, telegram_chat, telegram_enabled,
                    telegram_commands_enabled, telegram_live_preview
                )
                set_app_state("telegram_connection_enabled", "1")
                send_telegram_message(
                    telegram_token.strip(), telegram_chat.strip(),
                    f"TAO QUANT LAB V11 connected successfully to @{bot.get('username', 'bot')}. "
                    "I will keep reconnecting after temporary network failures until you press Disconnect."
                )
                st.success(f"Connected to @{bot.get('username', 'bot')}. Background monitoring is enabled.")
                st.rerun()
            except Exception as exc:
                set_app_state("telegram_worker_status", "CONNECTION FAILED")
                set_app_state("telegram_worker_error", str(exc)[:400])
                st.error(f"Telegram connection failed: {exc}")

    with disconnect_col:
        if st.button("Disconnect", use_container_width=True, disabled=not connection_requested):
            # Keep credentials and preferences saved; only stop polling and notifications.
            set_app_state("telegram_connection_enabled", "0")
            set_app_state("telegram_worker_status", "DISCONNECTING")
            st.success("Telegram disconnected. Your saved token and chat ID are retained.")
            st.rerun()

    if st.button("Forget saved Telegram credentials", use_container_width=True):
        set_app_state("telegram_connection_enabled", "0")
        clear_telegram_settings()
        for key in ("telegram_token_input", "telegram_chat_input"):
            st.session_state[key] = ""
        st.rerun()

    worker_status = get_app_state("telegram_worker_status", "NOT STARTED")
    worker_error = get_app_state("telegram_worker_error", "")
    last_ok = get_app_state("telegram_last_success", "")
    last_poll = get_app_state("telegram_last_poll_success", "")
    if connection_requested:
        st.success(f"Telegram connection requested · worker: {worker_status}")
    else:
        st.warning(f"Telegram is disconnected · worker: {worker_status}")
    if last_ok:
        st.caption(f"Last successful Telegram message: {format_ist(last_ok)}")
    if last_poll:
        st.caption(f"Last successful command poll: {format_ist(last_poll)}")
    if worker_error:
        st.error(f"Latest Telegram worker error: {worker_error}")
    st.caption(
        "V12 keeps retrying temporary Telegram/network failures and restores the connection after a Streamlit restart. "
        "The computer and Streamlit process must remain running; closing PowerShell or stopping the app ends background services."
    )

    if st.button("Check Telegram connection", use_container_width=True, disabled=not telegram_token.strip()):
        try:
            bot = get_telegram_bot_info(telegram_token.strip())
            webhook = get_telegram_webhook_info(telegram_token.strip())
            st.success(f"Telegram API reachable: @{bot.get('username', 'bot')}.")
            if webhook and webhook.get("url"):
                st.warning("Webhook configured. Connect/Reconnect will remove it for polling.")
            else:
                st.info("Webhook is clear.")
        except Exception as exc:
            st.error(str(exc))

    if st.button("Find my Telegram chat ID", use_container_width=True, disabled=not telegram_token.strip()):
        try:
            chats = discover_telegram_chats(telegram_token.strip())
            if chats:
                for item in chats[-5:]:
                    st.info(
                        f"Chat ID: {item['id']} · @{item['username'] or 'no username'} · "
                        f"Last message: {item['text'][:80]}"
                    )
            else:
                st.warning("Open your bot, press Start, send 'hello', then click this button again.")
        except Exception as exc:
            st.error(str(exc))

    st.header("Paper-trading assumptions")
    fee_bps = st.number_input("Fee per side (bps)", 0.0, 200.0, 10.0, 1.0)
    slip_bps = st.number_input("Slippage per side (bps)", 0.0, 300.0, 15.0, 1.0)
    set_app_state("paper_fee_bps", str(float(fee_bps)))
    set_app_state("paper_slippage_bps", str(float(slip_bps)))
    if st.button("↻ Refresh all market data", use_container_width=True):
        load_intraday.clear()
        load_daily.clear()
        st.rerun()
    st.caption("Historical indicator bars: Yahoo Finance 5-minute candles. Live execution reference: Binance Spot TAO/USDT WebSocket. USDT is not exactly USD.")
    st.caption("No exchange order execution is connected. The live feed is market-data only; it never places orders.")

try:
    with st.spinner("Loading TAO intraday and daily market data..."):
        intraday = load_intraday()
        daily_raw = load_daily(history)
    if daily_raw.empty:
        st.error("Daily TAO data unavailable. Check internet and refresh.")
        st.stop()
    daily = add_daily_indicators(daily_raw, float(flag_threshold))
except Exception as exc:
    st.error(f"Unable to load market data: {exc}")
    st.stop()

# Use the continuously streamed Binance spot quote for live marking; Yahoo is historical fallback only.
live_quote = get_live_quote()
if live_quote and live_quote.get("fresh"):
    reference_price = (float(live_quote["bid"])+float(live_quote["ask"])) / 2 if live_quote.get("bid") and live_quote.get("ask") else float(live_quote["price"])
    price_timestamp = live_quote["event_time"]
    price_source = "Binance Spot WebSocket · TAO/USDT"
elif not intraday.empty:
    reference_price = float(intraday["Close"].iloc[-1])
    price_timestamp = intraday.index[-1]
    price_source = "STALE FALLBACK · Yahoo 5-minute bar"
else:
    reference_price = float(daily["Close"].iloc[-1])
    price_timestamp = daily.index[-1]
    price_source = "STALE FALLBACK · Yahoo daily close"

last_daily = daily.iloc[-1]
daily_change = (
    float(daily["Close"].iloc[-1] / daily["Close"].iloc[-2] - 1)
    if len(daily) > 1 else np.nan
)
vol_forecast = last_daily["Persistence_30D"]
annualized_vol = (
    last_daily["Volatility_30D"] * np.sqrt(DAYS_PER_YEAR)
    if pd.notna(last_daily["Volatility_30D"]) else np.nan
)
signal = current_candidate_signal(last_daily)

# Portfolio marked to current available reference price.
p = get_portfolio()
trades = get_trade_history()
cash = float(p["cash"])
units = float(p["units"])
avg_cost = float(p["avg_cost"])
position_value = units * reference_price
equity = cash + position_value
unrealized = units * (reference_price - avg_cost)
total_pnl = equity - float(p["initial_cash"])
total_return = total_pnl / float(p["initial_cash"]) if p["initial_cash"] else 0

if auto_refresh:
    st_autorefresh(interval=1000, key="tao_live_dashboard_refresh")
if live_quote and live_quote.get("fresh"):
    st.success(f"🟢 LIVE STREAM · TAO/USDT ${reference_price:,.6f} · bid ${float(live_quote['bid'] or 0):,.6f} · ask ${float(live_quote['ask'] or 0):,.6f} · tick age {live_quote['age_seconds']:.2f}s")
else:
    st.error("🔴 LIVE FEED STALE / DISCONNECTED — do not rely on this fallback quote for trade decisions.")
st.caption(f"Reference: {money(reference_price)} · {price_source} · exchange event time: {format_ist(price_timestamp)}")

# Feed/Telegram service health.
feed_status = get_app_state("live_feed_status", "STARTING")
feed_detail = get_app_state("live_feed_detail", "")
health_a, health_b, health_c = st.columns(3)
health_a.metric("Market stream", feed_status)
health_b.metric("Live tick age", f"{live_quote['age_seconds']:.1f}s" if live_quote else "No ticks")
health_c.metric("Telegram worker", get_app_state("telegram_worker_status", "WAITING"))
if feed_detail and feed_status not in ("CONNECTED",): st.caption(f"Feed detail: {feed_detail}")

# Always-visible portfolio strip so the user sees holdings and TAO price.
st.subheader("Paper portfolio — current position")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("TAO/USDT reference", money(reference_price), percent(daily_change))
m2.metric("TAO held", f"{units:.6f}")
m3.metric("Position value", money(position_value))
m4.metric("Paper cash", money(cash))
m5.metric("Unrealized P/L", money(unrealized))

# Live chart uses the stream of actual exchange trades; Yahoo bars remain the historical analytics source.
st.subheader("TAO/USDT live price tape")
live_ticks = get_live_ticks(1200)
if not live_ticks.empty:
    live_ticks["event_time"] = pd.to_datetime(live_ticks["event_time"], utc=True, errors="coerce").dt.tz_convert(IST)
    live_ticks = live_ticks.dropna(subset=["event_time"]).set_index("event_time")
    chart = live_ticks["price"].tail(900)
    st.line_chart(chart.rename("TAO/USDT · exchange trades"), height=320)
    st.caption(f"Streaming Binance Spot trade ticks · {len(live_ticks):,} recent ticks stored locally. Quote currency is USDT, not USD.")
elif not intraday.empty:
    st.warning("Waiting for the first WebSocket tick. Showing Yahoo's last 5-minute bars as fallback.")
    st.line_chart(intraday["Close"].tail(288).rename("TAO historical 5-minute close"), height=300)
else:
    st.warning("No live ticks or intraday bars received yet.")

main_tabs = st.tabs([
    "Market watcher",
    "Trade monitor",
    "Analytics",
    "Signals & indicators",
    "Paper trading",
    "Performance & backtest",
    "Trade history",
    "Risk monitor",
])

# ------------------------------ MARKET WATCHER ------------------------------

with main_tabs[0]:
    st.header("🔎 TAO Market Watcher")
    st.caption("Scans closed Binance 1-minute candles plus the Yahoo daily trend when live candle history is available; falls back to Yahoo 5-minute candles during warm-up. Trade SL/TP is checked on every live trade tick.")
    live_signal_bars = get_live_candles(500, closed_only=True)
    signal_frame = live_signal_bars if len(live_signal_bars) >= 80 else intraday
    setup = score_market_setup(signal_frame, last_daily, int(alert_threshold))
    # Re-anchor illustrative risk levels to the live exchange quote, retaining ATR distances from the signal bar.
    if setup.get("direction") in ("LONG", "SHORT") and live_quote and live_quote.get("fresh"):
        old_price = float(setup["price"])
        # Use the executable side of the live book: ask for LONG, bid for SHORT.
        live_price = float(live_quote.get("ask") or live_quote["price"]) if setup["direction"] == "LONG" else float(live_quote.get("bid") or live_quote["price"])
        setup["stop"] = live_price + (float(setup["stop"])-old_price)
        setup["target"] = live_price + (float(setup["target"])-old_price)
        setup["price"] = live_price
    signal_ready = len(live_signal_bars) >= 80
    fresh_alert = record_alert_once(setup) if (live_quote and live_quote.get("fresh") and signal_ready) else False
    opened_signal_trade = False
    paper_entry_signature = market_signal_signature(setup) if setup.get("direction") in ("LONG", "SHORT") and setup.get("timestamp") is not None else ""
    if auto_paper and paper_entry_signature and live_quote and live_quote.get("fresh") and signal_ready:
        # V14: alert logging is separate from trade entry. One paper position is
        # allowed per direction and closed 1-minute signal candle; multiple
        # different signal candles can remain open concurrently.
        signal_ts = pd.Timestamp(setup["timestamp"])
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize(IST)
        else:
            signal_ts = signal_ts.tz_convert(IST)
        signal_age_seconds = max(0.0, (now_ist() - signal_ts.to_pydatetime()).total_seconds())
        if signal_age_seconds <= 180:
            opened_signal_trade = open_signal_trade(
                setup, float(paper_notional), float(fee_bps), float(slip_bps),
                entry_signature=paper_entry_signature,
            )
            if opened_signal_trade:
                update_alert_paper_status(paper_entry_signature, "OPENED", "Paper trade opened.")
            else:
                update_alert_paper_status(
                    paper_entry_signature, "NOT OPENED",
                    get_app_state("last_entry_block_reason", "Entry conditions were not met."),
                )
        else:
            update_alert_paper_status(paper_entry_signature, "EXPIRED", "Signal candle is older than 180 seconds; no late entry.")
    elif auto_paper and paper_entry_signature and not (live_quote and live_quote.get("fresh")):
        quote_age = live_quote.get("age_seconds", float("inf")) if live_quote else float("inf")
        conn_age = live_quote.get("connection_age_seconds", float("inf")) if live_quote else float("inf")
        pause_reason = f"PAUSED: Binance WebSocket updates are stale/disconnected (last trade tick age {quote_age:.1f}s; last book/feed update age {conn_age:.1f}s). No entry is allowed until a fresh book/feed update returns."
        update_alert_paper_status(paper_entry_signature, "PAUSED", pause_reason)
    elif auto_paper and paper_entry_signature and not signal_ready:
        update_alert_paper_status(paper_entry_signature, "PAUSED", "Waiting for at least 80 closed Binance 1-minute candles.")
    elif paper_entry_signature and not auto_paper:
        update_alert_paper_status(paper_entry_signature, "DISABLED", "Automatic signal paper-trading checkbox was turned off.")
    trade_exit = None  # SL/TP is monitored on every incoming exchange tick in the background feed thread.
    if auto_paper and setup["direction"] in ("LONG", "SHORT") and not (live_quote and live_quote.get("fresh")):
        age = live_quote.get("age_seconds", float("inf")) if live_quote else float("inf")
        received_age = live_quote.get("connection_age_seconds", float("inf")) if live_quote else float("inf")
        st.warning(f"PAPER ENTRY PAUSED — Binance WebSocket has not delivered a recent update (last trade tick {age:.1f}s ago; last book/feed update {received_age:.1f}s ago). This is a safety lock: no entry is opened on stale prices. When feed returns, only a still-fresh setup (max 180 seconds old) can be entered; older setups expire.")
    if auto_paper and setup["direction"] in ("LONG", "SHORT") and not signal_ready:
        st.warning("Automatic paper entry paused until at least 80 closed Binance 1-minute candles are available. Yahoo fallback signals are display-only.")
    if opened_signal_trade and telegram_enabled and telegram_token and telegram_chat:
        try:
            active_now = get_open_signal_trade()
            if active_now:
                queue_telegram_message(f"trade-open-{active_now['id']}",
                    f"🟢 TAO PAPER TRADE OPENED\nTrade #{active_now['id']} · {active_now['direction']}\nEntry: ${float(active_now['entry_price']):.4f}\nSL: ${float(active_now['stop']):.4f}\nTP: ${float(active_now['target']):.4f}\nNotional: ${float(active_now['notional']):.2f}\nMonitoring uses incoming exchange trade ticks. Paper simulation only.")
        except Exception as exc:
            st.warning(f"Telegram open-trade alert failed: {exc}")
    active_for_preview = get_open_signal_trade()
    if active_for_preview is None and get_app_state("telegram_preview_message_id", ""):
        clear_telegram_live_preview()
    telegram_status = get_app_state("telegram_worker_status", "NOT CONFIGURED")
    telegram_error = get_app_state("telegram_worker_error", "")
    st.caption(f"Telegram background worker: {telegram_status}" + (f" · {telegram_error}" if telegram_error else ""))

    if setup["direction"] == "LONG":
        st.success(f"🟢 POTENTIAL LONG SETUP · Score {setup['score']}/100")
    elif setup["direction"] == "SHORT":
        st.error(f"🔴 POTENTIAL SHORT SETUP · Score {setup['score']}/100")
    else:
        st.warning(f"🟡 WAIT — no qualifying setup · Best score {setup.get('score', 0)}/100")
        st.write(setup.get("reason", "Waiting for enough market data."))

    if setup.get("price"):
        w1,w2,w3,w4 = st.columns(4)
        w1.metric("TAO reference", money(setup["price"]))
        w2.metric("LONG score", f"{setup.get('long_score',0)}/100")
        w3.metric("SHORT score", f"{setup.get('short_score',0)}/100")
        w4.metric("RSI / ADX", f"{setup.get('rsi',np.nan):.1f} / {setup.get('adx',np.nan):.1f}")
        st.caption(f"Last evaluated bar: {format_ist(setup['timestamp'])} · Relative volume: {setup.get('relvol',np.nan):.2f}x · Bars: {'Binance 1-minute' if len(live_signal_bars) >= 80 else 'Yahoo 5-minute fallback'}")

    if setup["direction"] in ("LONG", "SHORT"):
        q1,q2,q3 = st.columns(3)
        q1.metric("Illustrative entry reference", money(setup["price"]))
        q2.metric("ATR-based stop reference", money(setup["stop"]))
        q3.metric("ATR-based target reference", money(setup["target"]))
        st.info("A score is a rule-confluence measure, not a probability of profit. V14 requires at least 5/6 factors, trend + MACD + daily alignment, ADX >= 20, and price no more than 1.5 ATR from EMA20. Paper entry additionally requires >=3:1 estimated NET reward/risk after fees/slippage. Stop/target are simulations, not placed orders.")
        alert_text = (
            f"TAO QUANT LAB — POTENTIAL {setup['direction']} SETUP\\n"
            f"Score: {setup['score']}/100\\n"
            f"TAO reference: ${setup['price']:.4f}\\n"
            f"Illustrative stop: ${setup['stop']:.4f}\\n"
            f"Illustrative target: ${setup['target']:.4f}\\n"
            f"RSI: {setup['rsi']:.1f} | ADX: {setup['adx']:.1f} | Rel volume: {setup.get('relvol',np.nan):.2f}x\\n"
            "Research alert only — not a trade instruction."
        )
        if fresh_alert:
            st.toast(f"New potential {setup['direction']} setup detected", icon="🔔")
            if enable_sound:
                st.components.v1.html("""<script>try { const C=window.AudioContext||window.webkitAudioContext; const c=new C(); const o=c.createOscillator(); const g=c.createGain(); o.connect(g); g.connect(c.destination); o.frequency.value=880; g.gain.value=0.08; o.start(); o.stop(c.currentTime+0.35); } catch(e) {}</script>""", height=0)
            if telegram_enabled and telegram_token and telegram_chat:
                try:
                    queue_telegram_message(f"market-alert-{market_signal_signature(setup)}", alert_text.replace('\\n','\n'))
                    st.success("New setup queued for Telegram delivery.")
                except Exception as exc:
                    st.error(f"Telegram alert failed: {exc}")
            elif telegram_enabled:
                st.warning("Telegram is enabled, but bot token or chat ID is missing.")
    else:
        st.caption("No qualifying LONG/SHORT alert at this time. WAIT is a valid result; the watcher is designed to avoid forcing a trade.")

    if opened_signal_trade:
        st.success(f"Automatic paper {setup['direction']} trade opened for ${float(paper_notional):,.0f} notional. Wider ATR-based SL/TP monitoring is active.")
    elif auto_paper and setup["direction"] in ("LONG", "SHORT"):
        block_reason = get_app_state("last_entry_block_reason", "")
        if block_reason:
            st.warning(block_reason)
        elif len(get_open_signal_trades(100)) >= int(get_app_state("max_concurrent_paper_trades", "5") or 5):
            st.info("Signal recorded, but the configured maximum number of simultaneous paper positions is already open.")

    st.info(
        "V14 paper policy: target = 3.5R gross; minimum estimated net reward/risk = 3.0:1 after configured fees/slippage; "
        "at least 5/6 factors with trend + MACD + daily alignment; ADX >= 20; no >1.5 ATR chase. "
        "Each unique qualifying 1-minute signal can open one paper position, up to the configured concurrent-position limit. "
        "Every open position is monitored separately on trade ticks and completed 1-minute candles."
    )

    active = get_open_signal_trade()
    if active:
        st.markdown("#### Active monitored paper trade")
        t1,t2,t3,t4,t5 = st.columns(5)
        t1.metric("Direction", active["direction"])
        t2.metric("Entry", money(active["entry_price"]))
        t3.metric("Stop loss", money(active["stop"]))
        t4.metric("Take profit", money(active["target"]))
        elapsed = max(0, (now_ist() - pd.Timestamp(active["entry_time"])).total_seconds()/60)
        t5.metric("Time open", f"{elapsed:.0f} min")
        st.caption("Every open position is checked on each Binance trade tick and again on each completed 1-minute candle. If both stop and target occur inside one candle, the simulation conservatively records the stop first.")

    st.markdown("#### Current factor checks")
    factors = pd.DataFrame([
        ["Intraday trend alignment", "EMA20/EMA50 + price alignment", "Included in score"],
        ["Momentum confirmation", "MACD versus signal line", "Included in score"],
        ["RSI zone", "52–68 long / 32–48 short", "Included in score"],
        ["Trend strength", "ADX ≥ 18", "Included in score"],
        ["Relative volume", "Latest volume ≥ 20-bar average", "Included in score"],
        ["Daily trend alignment", "Daily SMA20 versus SMA50", "Included in score"],
    ], columns=["Factor", "Rule", "Purpose"])
    st.dataframe(factors, hide_index=True, use_container_width=True)
    st.markdown("#### Recent alert log")
    alerts_df = get_market_alerts()
    if alerts_df.empty:
        st.info("No qualifying alert has been recorded yet.")
    else:
        alerts_display = records_in_ist(alerts_df.drop(columns=["signature"]))
        alerts_export = records_in_ist(alerts_df)
        preferred_alert_cols = [c for c in [
            "id","timestamp","direction","score","price","stop","target","status",
            "paper_trade_status","paper_trade_reason"
        ] if c in alerts_display.columns]
        st.dataframe(alerts_display[preferred_alert_cols], hide_index=True, use_container_width=True)
        st.download_button("Download alert history CSV (IST)", alerts_export.to_csv(index=False).encode("utf-8"), "tao_market_alerts_IST.csv", "text/csv")

    st.markdown("#### Telegram setup")
    st.write("1. Create a bot with @BotFather. 2. Open your bot and press Start, then send 'hello'. 3. Enter the bot token. 4. Click 'Find my Telegram chat ID' in the sidebar and copy the number into Chat ID. 5. Click 'Check Telegram connection'. 6. Enable alerts and live preview. Leave this dashboard running.")
    st.info("Commands: /last, /status, /live, /help. Commands and live preview run in a background long-poll worker and no longer depend on Streamlit page refreshes.")
    if st.button("Send Telegram test message", disabled=not (telegram_token and telegram_chat)):
        try:
            send_telegram_message(telegram_token, telegram_chat, "TAO QUANT LAB test: Telegram notifications are connected.")
            st.success("Test message sent. Check Telegram.")
        except Exception as exc:
            st.error(f"Telegram test failed: {exc}")

    st.caption("Sound may be blocked by browser autoplay settings until you interact with the page. Dashboard alerts and Telegram are separate notification paths.")
    with st.expander("Telegram troubleshooting"):
        st.markdown("- Send **/start** or **hello** to your bot first.\n- Confirm the chat ID is the numeric ID returned by 'Find my Telegram chat ID'.\n- Click 'Check Telegram connection'; remove any configured webhook with 'Repair polling'.\n- Only one running copy of this app should poll a bot token at a time.\n- Telegram commands are handled by a background long-poll worker; keep this Streamlit process running.\n- Telegram live preview requires an open paper trade and is not a second-by-second exchange feed.")

# ------------------------------ TRADE MONITOR ------------------------------

with main_tabs[1]:
    st.header("📍 Signal trade monitor")
    st.caption("Every qualifying fresh signal is paper-traded once per 1-minute signal candle, subject to the 3:1 net R:R and execution-safety checks. LONG and SHORT are simulated, not exchange positions.")
    active_trades = get_open_signal_trades(100)
    if active_trades:
        st.markdown(f"#### Open positions being monitored ({len(active_trades)})")
        open_view = pd.DataFrame(active_trades)
        open_view = records_in_ist(open_view)
        open_cols = [c for c in ["id","direction","entry_time","entry_price","stop","target","notional","signal_score","initial_reward_risk"] if c in open_view.columns]
        st.dataframe(open_view[open_cols], use_container_width=True, hide_index=True)
        active_trade = active_trades[0]
        direction = active_trade["direction"]
        entry = float(active_trade["entry_price"])
        qty = float(active_trade["quantity"])
        mark = reference_price
        gross_open = qty * (mark-entry) if direction == "LONG" else qty * (entry-mark)
        entry_fee = float(active_trade["notional"]) * float(fee_bps) / 10000
        est_exit_fee = qty * mark * float(fee_bps) / 10000
        est_net = gross_open - entry_fee - est_exit_fee
        elapsed = max(0.0, (now_ist()-pd.Timestamp(active_trade["entry_time"])).total_seconds()/60)
        st.success(f"OPEN {direction} · monitoring stop loss and take profit")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Entry price", money(entry))
        c2.metric("Current mark", money(mark))
        c3.metric("Stop loss", money(active_trade["stop"]))
        c4.metric("Take profit", money(active_trade["target"]))
        c5.metric("Elapsed", f"{elapsed:.0f} min")
        st.metric("Estimated open P/L after assumed fees (before exit slippage)", money(est_net))
        st.caption(f"Live exchange mark: {format_ist(price_timestamp)} · {price_source}. SL/TP checks run on each incoming trade tick.")
        progress = ((mark-entry)/(float(active_trade["target"])-entry) if direction == "LONG" else (entry-mark)/(entry-float(active_trade["target"])))
        st.progress(float(np.clip(progress, 0, 1)), text="Progress toward target (visual guide only)")
        st.info("Exit detection runs on Binance trade ticks and completed 1-minute candles. If the WebSocket disconnects, monitoring pauses until data resumes; the outage is shown in feed health.")
    else:
        st.info("No open signal paper trade. The watcher will open one automatically when a fresh qualifying LONG/SHORT setup appears and automatic paper-trading is enabled.")
    history_signal = get_signal_trade_history()
    st.markdown("#### Completed and active signal trades")
    if history_signal.empty:
        st.write("No signal trades recorded yet.")
    else:
        display = records_in_ist(history_signal)
        for col in ["entry_time", "exit_time"]:
            if col in display.columns:
                display[col] = display[col].fillna("—")
        st.dataframe(display, use_container_width=True, hide_index=True)
        closed = history_signal[history_signal["status"] == "CLOSED"].copy()
        if not closed.empty:
            wins = int((closed["net_pnl"] > 0).sum())
            z1,z2,z3,z4 = st.columns(4)
            z1.metric("Closed trades", len(closed))
            z2.metric("Winners", wins)
            z3.metric("Win rate", f"{wins/len(closed)*100:.1f}%")
            z4.metric("Net simulated P/L", money(closed["net_pnl"].sum()))
        st.download_button("Download signal trade log CSV (IST)", records_in_ist(history_signal).to_csv(index=False).encode("utf-8"), "tao_signal_trade_log_IST.csv", "text/csv")
    st.markdown("#### Learning from completed trades")
    if history_signal.empty:
        st.info("No completed trades yet. V14 records setup score, ATR, initial stop distance, reward/risk, exit reason, net P/L and duration for each new paper trade.")
    else:
        closed = history_signal[history_signal["status"] == "CLOSED"].copy()
        if closed.empty:
            st.info("No closed trades yet. Keep collecting paper-trade outcomes before changing the rules.")
        else:
            closed["net_pnl"] = pd.to_numeric(closed["net_pnl"], errors="coerce")
            closed["duration_minutes"] = pd.to_numeric(closed["duration_minutes"], errors="coerce")
            wins_df = closed[closed["net_pnl"] > 0]
            losses_df = closed[closed["net_pnl"] <= 0]
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Average net P/L", money(closed["net_pnl"].mean()))
            l2.metric("Average duration", f"{closed['duration_minutes'].mean():.1f} min")
            l3.metric("SL exits", int((closed["exit_reason"] == "STOP LOSS").sum()))
            l4.metric("TP exits", int((closed["exit_reason"] == "TAKE PROFIT").sum()))
            if "initial_risk_pct" in closed.columns and closed["initial_risk_pct"].notna().any():
                st.caption(f"Average planned stop distance in V14 trades: {closed['initial_risk_pct'].dropna().mean():.2f}% of entry.")
            st.caption(
                f"Observed sample: {len(closed)} closed trades, {len(wins_df)} winners and {len(losses_df)} non-winners. "
                "These are descriptive results, not proof of a durable edge. Do not auto-adjust parameters from a tiny sample."
            )
            available_cols = [c for c in [
                "id","direction","signal_score","entry_price","stop","target","initial_risk_pct",
                "initial_reward_risk","exit_reason","net_pnl","duration_minutes"
            ] if c in closed.columns]
            st.dataframe(closed[available_cols].tail(30), use_container_width=True, hide_index=True)
    st.caption("If the app is closed, it cannot monitor prices. All open positions are monitored from Binance trade ticks plus completed 1-minute candles while the app is running. A feed outage pauses monitoring until reconnect; gaps cannot be reconstructed perfectly.")


# ------------------------------ ANALYTICS ------------------------------

with main_tabs[2]:
    st.header("Market analytics")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("RSI (14)", "—" if pd.isna(last_daily["RSI_14"]) else f'{last_daily["RSI_14"]:.1f}')
    a2.metric("14-day momentum", percent(last_daily["Momentum_14D"]))
    a3.metric("30-day volatility (annualized)", percent(annualized_vol))
    a4.metric("Next-day absolute-return estimate", percent(vol_forecast))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Daily price with trend averages")
        st.line_chart(
            daily[["Close", "SMA_20", "SMA_50"]].tail(365),
            height=300,
        )
    with right:
        st.markdown("#### Daily return (%)")
        st.bar_chart(
            (daily["Return"].tail(180) * 100).rename("Daily return (%)"),
            height=300,
        )

    st.markdown("#### Volatility history")
    vol_chart = pd.DataFrame({
        "7-day volatility": daily["Volatility_7D"] * np.sqrt(DAYS_PER_YEAR) * 100,
        "14-day volatility": daily["Volatility_14D"] * np.sqrt(DAYS_PER_YEAR) * 100,
        "30-day volatility": daily["Volatility_30D"] * np.sqrt(DAYS_PER_YEAR) * 100,
    }).tail(365)
    st.line_chart(vol_chart, height=280)

    st.markdown("#### Recent market data")
    recent = daily[[
        "Close", "Raw_Return", "Momentum_14D", "RSI_14",
        "Volatility_30D", "Persistence_30D"
    ]].tail(15).sort_index(ascending=False)
    st.dataframe(recent, use_container_width=True)

    st.info(
        "The 30-day persistence figure estimates the size of tomorrow's absolute return, "
        "not its direction. It is a volatility proxy, not a buy/sell forecast."
    )

# ------------------------------ SIGNALS ------------------------------

with main_tabs[3]:
    st.header("Signal and indicator analysis")
    if signal == "BUY CANDIDATE":
        st.success("BUY CANDIDATE — experimental rule conditions are currently met.")
    elif signal == "CASH / EXIT CANDIDATE":
        st.warning("CASH / EXIT CANDIDATE — experimental bearish conditions are met.")
    else:
        st.info(f"Current rule status: {signal}")

    st.markdown("#### Rule conditions")
    conditions = {
        "Close above SMA 50": bool(last_daily["Close"] > last_daily["SMA_50"]) if pd.notna(last_daily["SMA_50"]) else False,
        "SMA 20 above SMA 50": bool(last_daily["SMA_20"] > last_daily["SMA_50"]) if pd.notna(last_daily["SMA_20"]) and pd.notna(last_daily["SMA_50"]) else False,
        "RSI between 50 and 70": bool(50 <= last_daily["RSI_14"] < 70) if pd.notna(last_daily["RSI_14"]) else False,
        "MACD above signal": bool(last_daily["MACD"] > last_daily["MACD_Signal"]) if pd.notna(last_daily["MACD"]) and pd.notna(last_daily["MACD_Signal"]) else False,
    }
    st.dataframe(
        pd.DataFrame({
            "Condition": list(conditions.keys()),
            "Currently true": ["YES" if v else "NO" for v in conditions.values()]
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("#### Indicator charts")
    st.line_chart(daily[["RSI_14"]].tail(180), height=220)
    st.line_chart(daily[["MACD", "MACD_Signal"]].tail(180), height=220)
    st.caption(
        "These are transparent experimental rules, not validated trading signals. "
        "A candidate label does not mean the trade has positive expected value."
    )

# ------------------------------ PAPER TRADING ------------------------------

with main_tabs[4]:
    st.header("Paper trading")
    st.warning(
        "SIMULATION ONLY — these buttons update your local virtual portfolio. "
        "They do not connect to an exchange or send real orders."
    )
    st.markdown("#### Your open TAO position")
    pos1, pos2, pos3, pos4 = st.columns(4)
    pos1.metric("TAO quantity", f"{units:.8f}")
    pos2.metric("Average cost", money(avg_cost) if units > 0 else "No open position")
    pos3.metric("Current reference price", money(reference_price))
    pos4.metric("Unrealized P/L", money(unrealized))

    if units > 0:
        st.success(
            f"You currently hold {units:.8f} TAO. "
            f"Estimated market value: {money(position_value)}."
        )
    else:
        st.info("You currently hold 0 TAO in the paper portfolio. Use the Buy panel below to simulate a position.")

    buy_col, sell_col = st.columns(2)
    with buy_col:
        st.markdown("### Buy TAO")
        spend = st.number_input(
            "USD amount to spend",
            min_value=1.0,
            max_value=max(1.0, cash),
            value=min(100.0, max(1.0, cash)),
            step=10.0,
            key="v4_buy_spend",
        )
        st.caption(
            f"Reference price {money(reference_price)} · assumed fee {fee_bps:.1f} bps · "
            f"assumed slippage {slip_bps:.1f} bps"
        )
        if st.button("Simulate paper BUY", type="primary", use_container_width=True):
            try:
                qty, fill, fee = paper_buy(spend, reference_price, fee_bps, slip_bps)
                st.success(
                    f"Bought {qty:.8f} TAO at simulated {money(fill)}. Fee: {money(fee)}."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    with sell_col:
        st.markdown("### Sell TAO")
        if units > 0:
            qty_to_sell = st.number_input(
                "TAO quantity to sell",
                min_value=0.000001,
                max_value=float(units),
                value=float(units),
                step=0.01,
                format="%.8f",
                key="v4_sell_qty",
            )
        else:
            qty_to_sell = 0.0
            st.number_input(
                "TAO quantity to sell",
                min_value=0.000001,
                value=0.000001,
                disabled=True,
                key="v4_sell_disabled",
            )
        st.caption("Sell quantity is limited to the TAO held in your simulated account.")
        if st.button(
            "Simulate paper SELL",
            use_container_width=True,
            disabled=(units <= 0),
        ):
            try:
                gross, fill, fee, realized = paper_sell(
                    qty_to_sell, reference_price, fee_bps, slip_bps
                )
                st.success(
                    f"Sold at simulated {money(fill)}. Gross {money(gross)}, "
                    f"fee {money(fee)}, realized P/L {money(realized)}."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("#### Portfolio value")
    st.metric("Total paper account equity", money(equity), f"{total_return * 100:+.2f}% since start")
    st.caption(
        "The portfolio uses the latest available Yahoo bar as its mark price. "
        "The bar may be delayed; fills are hypothetical and include the assumed costs."
    )

# ------------------------------ BACKTEST ------------------------------

with main_tabs[5]:
    st.header("Strategy performance and backtest")
    st.write(
        "This tests the experimental long/cash technical rule. It calculates the signal "
        "at a daily close and applies the position to the following daily return. "
        "Fees and slippage are charged when the position changes."
    )
    bt = backtest_long_cash(daily, float(fee_bps), float(slip_bps))
    if len(bt) < 60:
        st.warning("Not enough valid daily observations for this backtest.")
    else:
        s = stats(bt["Strategy_Return"], bt["Strategy_Equity"])
        h = stats(bt["BuyHold_Return"], bt["BuyHold_Equity"])
        st.markdown("#### Strategy versus buy-and-hold")
        left, right = st.columns(2)
        with left:
            st.markdown("**Experimental long/cash rule**")
            for key, val in s.items():
                if key == "Sharpe (rough)":
                    st.metric(key, "—" if pd.isna(val) else f"{val:.2f}")
                else:
                    st.metric(key, percent(val))
        with right:
            st.markdown("**TAO buy-and-hold reference**")
            for key, val in h.items():
                if key == "Sharpe (rough)":
                    st.metric(key, "—" if pd.isna(val) else f"{val:.2f}")
                else:
                    st.metric(key, percent(val))
        st.line_chart(
            bt[["Strategy_Equity", "BuyHold_Equity"]],
            height=320,
        )
        exposure = bt["Position"].mean() * 100
        entries = int((bt["Position"].diff().fillna(bt["Position"]) > 0).sum())
        exits = int((bt["Position"].diff().fillna(0) < 0).sum())
        z1, z2, z3 = st.columns(3)
        z1.metric("Time invested", f"{exposure:.1f}%")
        z2.metric("Entries", str(entries))
        z3.metric("Exits", str(exits))
        st.caption(
            "This is a preliminary historical test, not proof of an edge. It does not fully "
            "model spread, market impact, taxes, funding, exchange outages or order execution. "
            "The hand-built rule was selected after looking at historical data, so results may "
            "be optimistic. Validate on a genuinely untouched future period before considering live use."
        )

# ------------------------------ TRADE HISTORY ------------------------------

with main_tabs[6]:
    st.header("Paper trade history")
    history_df = get_trade_history()
    if history_df.empty:
        st.info("No simulated trades yet. Buy or sell TAO in the Paper trading tab.")
    else:
        history_display = records_in_ist(history_df)
        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download trade history CSV (IST)",
            data=history_display.to_csv(index=False).encode("utf-8"),
            file_name="tao_paper_trade_history_IST.csv",
            mime="text/csv",
        )

# ------------------------------ RISK ------------------------------

with main_tabs[7]:
    st.header("Portfolio risk monitor")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Starting balance", money(p["initial_cash"]))
    r2.metric("Current equity", money(equity))
    r3.metric("Realized P/L", money(p["realized_pnl"]))
    r4.metric("Total P/L", money(total_pnl))

    if units > 0:
        stress_pct = st.slider("Hypothetical TAO price fall", 1, 80, 10)
        stress_price = reference_price * (1 - stress_pct / 100)
        stress_equity = cash + units * stress_price
        st.metric("Hypothetical account equity after fall", money(stress_equity))
        st.metric("Hypothetical loss from current equity", money(equity - stress_equity))
        st.caption("Scenario only. This does not place a stop order or limit real losses.")
    else:
        st.info("No open TAO paper position. Risk scenario becomes available after a simulated buy.")

    st.divider()
    st.markdown("#### Reset paper portfolio")
    confirm = st.checkbox(
        "I understand that resetting permanently deletes this local paper-trade history."
    )
    if st.button("Reset portfolio to $10,000", disabled=not confirm):
        reset_paper_account()
        st.success("Paper portfolio reset.")
        st.rerun()

st.divider()
st.error(
    "RESEARCH AND PAPER TRADING ONLY. No real orders are placed. Intraday Yahoo data is "
    "not guaranteed real-time. Candidate signals and historical backtests are experimental "
    "and do not establish future profitability."
)
st.caption(
    f"TAO Quant Lab V6 · Local ledger: {DB_PATH.name} · "
    "Keep this script and database in the same folder to preserve your paper portfolio."
)
