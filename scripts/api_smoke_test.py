"""Local API smoke test; does not require model artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app


def main() -> int:
    client = TestClient(app)
    for path in ("/health", "/ready", "/status"):
        response = client.get(path)
        print(path, response.status_code, response.json())
        if response.status_code != 200:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
