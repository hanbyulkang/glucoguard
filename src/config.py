"""Project-wide constants. Everything that a reviewer might want to change lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"

ARCHIVE_NAME = "n=240-April-2026-OpenAPS-Data-Commons-unzipped-JSON.zip"


def _find_archive():
    """Locate the OpenAPS archive without assuming where this repo sits.

    The project moved into a competition folder partway through, which silently
    broke a hard-coded relative path. Walking up from here and looking for
    `dataset/OpenAPS/` survives that, and an explicit `GLUCOGUARD_ARCHIVE`
    environment variable overrides it for anyone whose copy lives elsewhere.
    """
    import os

    override = os.environ.get("GLUCOGUARD_ARCHIVE")
    if override:
        return Path(override)

    for base in [ROOT, *ROOT.parents]:
        candidate = base / "dataset" / "OpenAPS" / ARCHIVE_NAME
        if candidate.exists():
            return candidate
    return ROOT.parent / "dataset" / "OpenAPS" / ARCHIVE_NAME


OPENAPS_ZIP = _find_archive()

# --- CGM signal ---------------------------------------------------------------
SAMPLE_MINUTES = 5          # OpenAPS/Dexcom native cadence
GLUCOSE_MIN = 20            # mg/dL; below this the sensor reports an error code
GLUCOSE_MAX = 450           # mg/dL; Dexcom G4/G5 saturate at 400

# --- Forecasting task ---------------------------------------------------------
HISTORY_STEPS = 24          # 24 * 5 min = 2 h of context
HORIZON_MINUTES = 30
HORIZON_STEPS = HORIZON_MINUTES // SAMPLE_MINUTES   # predict t+30 min

# A window is only usable if the gap it was interpolated across is short enough.
MAX_INTERPOLATION_GAP_STEPS = 3     # 15 min; longer gaps invalidate the window

# --- Clinical thresholds ------------------------------------------------------
# ADA/ATTD international consensus on CGM metrics (Battelino et al., Diabetes Care 2019).
HYPO_THRESHOLD = 70         # mg/dL — level 1 hypoglycaemia
HYPER_THRESHOLD = 180       # mg/dL — level 1 hyperglycaemia

# --- Splits -------------------------------------------------------------------
SEED = 1337
TRAIN_FRAC, VAL_FRAC = 0.65, 0.15   # remainder is test; split is by PATIENT, not row
