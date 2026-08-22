"""DocSentry backend package.

Loads backend/.env for local development. Real environment variables always
win (override=False), so a stale local file cannot shadow what Render or
Spaces injects, and a missing file is not an error.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
