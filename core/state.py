def get_rhythm_state(chunk: dict, beat_interval_ms: float) -> str:
    score       = chunk.get("score", 0.0)
    drift_label = chunk.get("drift_label", "UNKNOWN")
    onset_count = chunk.get("onset_count", 0)
    beat_count  = chunk.get("beat_count", 0)

    drift_ms = 0.0
    if "LATE" in drift_label:
        try:
            drift_ms = float(drift_label.split("+")[-1].replace("ms)", "").replace("ms", ""))
        except ValueError:
            drift_ms = 100.0
    elif "EARLY" in drift_label:
        try:
            drift_ms = -abs(float(drift_label.split("-")[-1].replace("ms)", "").replace("ms", "")))
        except ValueError:
            drift_ms = -100.0

    beat_ratio = (beat_count / onset_count) if onset_count > 0 else 1.0

    if score >= 0.80 and abs(drift_ms) <= 80:
        return "GOOD"
    elif drift_ms < -80:
        return "EARLY"
    elif drift_ms > 80:
        return "LATE"
    elif beat_ratio > 1.3:
        return "FAST"
    else:
        return "SLOW"
