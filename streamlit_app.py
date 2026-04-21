# Punkt wejścia Streamlit Community Cloud — UI w app.py (ten plik tylko deleguje).

from __future__ import annotations

from pathlib import Path
import runpy

_APP = Path(__file__).resolve().parent / "app.py"
runpy.run_path(str(_APP), run_name="__main__")
