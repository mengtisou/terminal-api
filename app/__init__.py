"""Package init.

Loads .env before ANY other module imports, so os.getenv() is reliable
everywhere regardless of import order. Without this, app.providers can be
imported before app.config runs load_dotenv() and every key looks missing.
"""
from pathlib import Path

try:
    from dotenv import load_dotenv

    _ENV = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_ENV, override=False)
except ImportError:  # python-dotenv not installed - rely on real env vars
    pass
