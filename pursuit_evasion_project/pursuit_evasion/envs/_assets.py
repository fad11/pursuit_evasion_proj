import sys
from pathlib import Path


def resolve_asset(filename):
    source_asset = Path(__file__).resolve().parents[2] / "assets" / filename
    installed_asset = Path(sys.prefix) / "pursuit_evasion" / "assets" / filename
    for path in (source_asset, installed_asset):
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Could not find required simulator asset: {filename}")
