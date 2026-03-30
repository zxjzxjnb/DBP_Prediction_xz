"""Launch the Streamlit workbench from a normal console script."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Start the Streamlit app."""
    try:
        from streamlit.web import cli as stcli
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install the UI extras first, for example:\n"
            'pip install -e ".[ui]"'
        ) from exc

    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(stcli.main())
