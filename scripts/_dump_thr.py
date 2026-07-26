import json, numpy as np
from src.alarm_policy import tune_event_threshold, rolling_thresholds
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_calibration import scores_for
from scripts.eval_external import external_windows

TARGET, WARMUP = 6.0, 14.0
name = json.loads((ARTIFACTS_DIR/"selection.json").read_text())["selected"]
fc = Forecaster(name)
w = build_windows(verbose=False)

shared = tune_event_threshold(w["val"].y, scores_for(fc, w["val"].X), TARGET)
print(f"공용 임계값 (validation 기준): {shared:.4f}  = {shared:.1%}\n")

for cohort, ws in {"test": w["test"], "external": external_windows()}.items():
    score = scores_for(fc, ws.X)
    fixed, rolling_all = [], []
    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        wu = split_by_time(t, WARMUP)
        if int((y[wu] < HYPO_THRESHOLD).sum()) < 20: continue
        f = tune_event_threshold(y[wu], s[wu], TARGET)
        fixed.append(f)
        thr, _ = rolling_thresholds(y, s, t, TARGET)
        rolling_all.append(thr[np.isfinite(thr)])
    fixed = np.array(fixed); roll = np.concatenate(rolling_all)
    print(f"=== {cohort} ({len(fixed)}명) ===")
    print(f"  2주 고정  중앙값 {np.median(fixed):.1%}  범위 {fixed.min():.1%} ~ {fixed.max():.1%}")
    print(f"  롤링      중앙값 {np.median(roll):.1%}  범위 {roll.min():.1%} ~ {roll.max():.1%}")
    print(f"  환자별 고정값: {', '.join(f'{v:.1%}' for v in np.sort(fixed))}")
    print(f"  저혈당 기저율: {(ws.y < HYPO_THRESHOLD).mean():.2%}\n")
