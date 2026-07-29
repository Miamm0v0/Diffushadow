"""Path resolution for the campaign tree.

All analysis scripts locate data by *filename* rather than by hard-coded
directory, so the tree can be reorganised without breaking them.
"""
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent.parent
FIGURES = CAMPAIGN / "figures"
REPORTS = CAMPAIGN / "reports"
REFS = CAMPAIGN / "refs"
RESULTS_1D = CAMPAIGN / "results_1d"
RESULTS_2D = CAMPAIGN / "results_2d"


def find(name: str) -> Path:
    """Locate a data file anywhere under campaign/ by filename."""
    hits = sorted(CAMPAIGN.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name!r} not found under {CAMPAIGN}")
    return hits[0]


def find_all(pattern: str):
    """All files under campaign/ matching a glob pattern."""
    return sorted(CAMPAIGN.rglob(pattern))
