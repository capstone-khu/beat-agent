import sys
print("실행 python:", sys.executable)

import utils.compat  # Python/NumPy 호환성 패치 (반드시 먼저 임포트)

import json

from config import TOLERANCE_MS
from agents.violin_rhythm_agent import ViolinRhythmAgent
from core.beat_utils import evaluate_beat_detection
from utils.evaluation import evaluate_performance
from utils.visualization import plot_beat_comparison

if __name__ == "__main__":
    MIDI_PATH            = "twinkle2.mid"
    AUDIO_PATH           = "./audio/performance2.mp4"
    SCORE_METADATA_PATH  = "score_metadata.json"   # None으로 바꾸면 beat_grid 방식으로 fallback

    agent = ViolinRhythmAgent(MIDI_PATH, score_metadata_path=SCORE_METADATA_PATH)
    res, aligned_audio_start, beat_times_final, beat_grid_final, raw_beat_times_final = \
        agent.process(AUDIO_PATH)

    shifted = beat_grid_final + aligned_audio_start
    print("\n[진단] Beat Grid vs madmom 첫 10개:")
    for i in range(min(10, len(shifted), len(beat_times_final))):
        diff = (beat_times_final[i] - shifted[i]) * 1000
        print(f"  [{i:2d}] grid={shifted[i]:.3f}s  "
              f"madmom={beat_times_final[i]:.3f}s  diff={diff:+.0f}ms")

    print(f"\n[진단]  Beat Grid {len(beat_grid_final)}개 "
          f"({shifted[0]:.3f}~{shifted[-1]:.3f}s)")
    print(f"        madmom   {len(beat_times_final)}개 "
          f"({beat_times_final[0]:.3f}~{beat_times_final[-1]:.3f}s)")

    detection_stats = evaluate_beat_detection(
        beat_grid    = beat_grid_final,
        beat_times   = beat_times_final,
        audio_start  = aligned_audio_start,
        tolerance_ms = TOLERANCE_MS,
    )
    summary = evaluate_performance(res)
    print("\n=== 전체 성과 평가 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    plot_beat_comparison(
        beat_grid       = beat_grid_final,
        beat_times      = beat_times_final,
        audio_start     = aligned_audio_start,
        detection_stats = detection_stats,
        title           = "twinkle - Beat Grid vs madmom (BPM 리샘플)",
    )

    SUPERVISOR = None   # 슈퍼바이저 객체 (미구현 — 연결 시 교체)
    reports = agent.run_agent(
        json_results = res,
        supervisor   = SUPERVISOR,
    )

    print("\n=== 에이전트 보고 요약 ===")
    good_count = sum(1 for r in reports if r["state"] == "GOOD")
    print(f"  총 chunk: {len(reports)}개  GOOD: {good_count}개  "
          f"비율: {good_count/len(reports)*100:.1f}%")
    print("\n=== 최종 Q-Table ===")
    print(agent.q_table.summary())
