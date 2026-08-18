"""Central configuration. Everything tunable lives here."""
import os
from dataclasses import dataclass, field


# --- Model routing -----------------------------------------------------------
# Provider selection lives in app/llm.py and is driven entirely by .env:
#   REASONING_CHAIN=gemini,anthropic
#   ANTHROPIC_REASONING_MODEL=claude-sonnet-5
# Nothing here needs editing to switch providers.

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Optional data providers
BINANCE_BASE = "https://api.binance.com"
OANDA_TOKEN = os.getenv("OANDA_TOKEN", "")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT", "")
OANDA_BASE = os.getenv("OANDA_BASE", "https://api-fxpractice.oanda.com")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")


@dataclass
class RiskRules:
    """Hard gates applied to every generated signal, in code, after the model."""

    min_risk_reward: float = 1.5
    # Entry must sit within this many ATRs of the last price.
    max_entry_distance_atr: float = 1.5
    # Stop must be at least this many ATRs away (no scalping inside noise)...
    min_stop_distance_atr: float = 0.5
    # ...and no further than this (no 200-pip stops on a 15m chart).
    max_stop_distance_atr: float = 4.0
    min_confidence: float = 0.45
    # Refuse to trade into a high-impact event inside this window.
    news_blackout_minutes: int = 60
    # Refuse if the last candle is older than this.
    max_data_age_seconds: int = 900


@dataclass
class Settings:
    risk: RiskRules = field(default_factory=RiskRules)
    # Candles pulled per request. 300 is enough for EMA200 + structure.
    candle_limit: int = 300
    max_news_in_context: int = 5
    news_lookback_hours: int = 48


settings = Settings()
