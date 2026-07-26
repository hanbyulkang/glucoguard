import json, numpy as np
from src.alarm_policy import tune_event_threshold, event_metrics
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_calibration import scores_for
from scripts.eval_external import external_windows

TARGET, WARMUP = 6.0, 14.0
fc = Forecaster(json.loads((ARTIFACTS_DIR/"selection.json").read_text())["selected"])
w = build_windows(verbose=False)
rows = {}
for cohort, ws in {"test": w["test"], "external": external_windows()}.items():
    score = scores_for(fc, ws.X); out=[]
    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        wu = split_by_time(t, WARMUP)
        lows_wu = int((y[wu] < HYPO_THRESHOLD).sum())
        base = float((y < HYPO_THRESHOLD).mean())
        if lows_wu < 20:
            out.append({"pid": pid, "thr": None, "base": base, "lows_wu": lows_wu}); continue
        thr = tune_event_threshold(y[wu], s[wu], TARGET)
        m = event_metrics(y[~wu], s[~wu] >= thr)
        out.append({"pid": pid, "thr": float(thr), "base": base, "lows_wu": lows_wu,
                    "recall": m.episode_recall, "fa": m.false_alarms_per_day})
    rows[cohort] = out
json.dump(rows, open(ARTIFACTS_DIR/"thresholds.json","w"), indent=2)

for cohort, out in rows.items():
    ok = [r for r in out if r["thr"] is not None]
    thrs = np.array([r["thr"] for r in ok])
    print(f"\n=== {cohort}, 개인 임계값 {len(ok)}명 (보정 실패 {len(out)-len(ok)}명) ===")
    print(f"  중앙값 {np.median(thrs):.1%}   범위 {thrs.min():.1%} ~ {thrs.max():.1%}   "
          f"사분위 {np.percentile(thrs,25):.1%} / {np.percentile(thrs,75):.1%}")
    print(f"  {'환자':<12}{'임계값':>9}{'저혈당 기저율':>14}{'배수':>7}{'경고율':>9}{'헛경보/일':>10}")
    for r in sorted(ok, key=lambda r: r["thr"]):
        print(f"  {r['pid']:<12}{r['thr']:>8.1%}{r['base']:>13.2%}"
              f"{r['thr']/r['base']:>7.1f}x{r['recall']:>9.1%}{r['fa']:>10.1f}")
    for r in out:
        if r["thr"] is None:
            print(f"  {r['pid']:<12}{'보정 불가':>9}{r['base']:>13.2%}  (warm-up 저혈당 {r['lows_wu']}건)")
