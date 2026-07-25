"""Project-wide constants. Everything that a reviewer might want to change lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
ARTIFACTS_DIR = ROOT / "artifacts"

OPENAPS_ZIP = (
    ROOT.parent
    / "dataset"
    / "OpenAPS"
    / "n=240-April-2026-OpenAPS-Data-Commons-unzipped-JSON.zip"
)

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
