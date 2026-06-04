import json

import numpy as np


def evaluate_performance(json_results: str) -> dict:
    data = json.loads(json_results)
    if not data:
        return {"error": "결과 없음"}
    valid = [c for c in data if c.get("onset_count", 2) >= 2]
    if not valid:
        valid = data
    timing_avg = float(np.mean([c.get("timing_score", c.get("score", 0.0)) for c in valid]))
    if timing_avg >= 0.80:
        level, recommend = "훌륭", "더 어려운 곡 도전 추천"
    elif timing_avg >= 0.60:
        level, recommend = "적정", "리듬 안정성 집중 연습 권장"
    else:
        level, recommend = "미흡", "느린 템포로 메트로놈 연습 권장"
    return {
        "overall_score":     round(timing_avg, 3),
        "performance_level": level,
        "recommendation":    recommend,
        "weaknesses":        ["박자 안정성 부족"] if timing_avg < 0.60 else [],
    }
