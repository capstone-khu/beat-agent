import json
from collections import defaultdict

import numpy as np
import pretty_midi
import madmom
import madmom.features.beats as mf_beats

from config import REALTIME_MODE, TOLERANCE_MS


def run_madmom(audio_path: str, midi_bpm: float) -> np.ndarray:
    min_bpm = midi_bpm * 0.95
    max_bpm = midi_bpm * 1.05
    print(f"[madmom] {'online' if REALTIME_MODE else 'offline'} 모드  "
          f"BPM 범위: {min_bpm:.1f}~{max_bpm:.1f}")
    act  = mf_beats.RNNBeatProcessor(online=REALTIME_MODE)(audio_path)
    proc = mf_beats.DBNBeatTrackingProcessor(
               fps=100, min_bpm=min_bpm, max_bpm=max_bpm)
    beats = proc(act)
    print(f"[madmom] 검출 beat: {len(beats)}개  "
          f"마지막: {beats[-1]:.3f}s  "
          f"평균 간격: {np.diff(beats).mean()*1000:.1f}ms")
    return beats


def build_beat_grid_from_midi(midi_path: str) -> np.ndarray:
    midi       = pretty_midi.PrettyMIDI(midi_path)
    beat_times = midi.get_beats()
    print(f"[Beat Grid] {len(beat_times)}개  "
          f"평균 간격: {np.diff(beat_times).mean()*1000:.1f}ms")
    return beat_times


def build_chunks_from_score_metadata(metadata_path: str) -> list:
    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    notes    = data["notes"]
    bpm_meta = data.get("bpm", None)

    if bpm_meta:
        beat_interval = 60.0 / bpm_meta
    else:
        durations = [n["duration"] for n in notes]
        beat_interval = float(np.median(durations))

    half_dur = beat_interval * 2

    measure_notes: dict = defaultdict(list)
    for n in notes:
        measure_notes[n["measure"]].append(n)

    sorted_measures = sorted(measure_notes.keys())
    chunks = []

    for i, m in enumerate(sorted_measures):
        ns      = sorted(measure_notes[m], key=lambda x: x["start"])
        m_start = ns[0]["start"]
        m_half  = m_start + half_dur

        if i + 1 < len(sorted_measures):
            next_m  = sorted_measures[i + 1]
            next_ns = sorted(measure_notes[next_m], key=lambda x: x["start"])
            m_end   = next_ns[0]["start"]
        else:
            m_end = ns[-1]["end"]

        front_notes = [n for n in ns if n["start"] <  m_half]
        back_notes  = [n for n in ns if n["start"] >= m_half]

        chunks.append({
            "measure":    m,
            "half":       1,
            "start":      round(m_start, 3),
            "end":        round(m_half,  3),
            "note_count": len(front_notes),
            "notes":      front_notes,
        })
        chunks.append({
            "measure":    m,
            "half":       2,
            "start":      round(m_half, 3),
            "end":        round(m_end,  3),
            "note_count": len(back_notes),
            "notes":      back_notes,
        })

    print(f"[Score Metadata] {metadata_path} 로드 완료")
    print(f"  마디 수: {len(sorted_measures)}  →  chunk 수: {len(chunks)}")
    print(f"  반마디 길이: {half_dur*1000:.0f}ms  (beat_interval={beat_interval*1000:.0f}ms)")
    return chunks


def align_to_grid(beat_grid, beat_times, audio_start, beat_interval) -> tuple:
    ref        = beat_grid + audio_start
    first_beat = beat_times[0]
    diffs      = np.abs(ref - first_beat)
    best_idx   = int(np.argmin(diffs))
    best_diff  = diffs[best_idx]
    if best_diff < beat_interval:
        new_start = first_beat - beat_grid[best_idx]
        print(f"[Align] madmom 첫 beat({first_beat:.3f}s) "
              f"→ grid[{best_idx}]({beat_grid[best_idx]:.3f}s) 매핑  "
              f"diff={best_diff*1000:.0f}ms")
    else:
        new_start = first_beat - beat_grid[0]
        print(f"[Align] fallback: first_beat - grid[0] = {new_start:.3f}s")
    print(f"[Align] audio_start: {audio_start:.3f}s → {new_start:.3f}s")
    return new_start, beat_times


def resample_to_midi_bpm(beat_times, beat_grid, audio_start,
                          beat_interval) -> np.ndarray:
    first_beat = beat_times[0]
    last_time  = beat_times[-1]
    n_beats    = int((last_time - first_beat) / beat_interval) + 1
    resampled  = np.array([first_beat + i * beat_interval for i in range(n_beats)])
    print(f"[BPM 리샘플] {len(beat_times)}개 → {len(resampled)}개  "
          f"간격: {beat_interval*1000:.1f}ms (MIDI BPM 고정)  "
          f"범위: {resampled[0]:.3f}~{resampled[-1]:.3f}s")
    return resampled


def evaluate_beat_detection(beat_grid, beat_times, audio_start,
                             tolerance_ms=TOLERANCE_MS) -> dict:
    tol       = tolerance_ms / 1000.0
    reference = beat_grid + audio_start
    matched_ref  = set()
    matched_pred = set()
    for pi, pred in enumerate(beat_times):
        diffs = np.abs(reference - pred)
        ri    = int(np.argmin(diffs))
        if diffs[ri] <= tol and ri not in matched_ref:
            matched_ref.add(ri)
            matched_pred.add(pi)
    n_matched     = len(matched_ref)
    n_ref, n_pred = len(reference), len(beat_times)
    precision = n_matched / n_pred if n_pred > 0 else 0.0
    recall    = n_matched / n_ref  if n_ref  > 0 else 0.0
    f_measure = (2*precision*recall/(precision+recall)
                 if (precision+recall) > 0 else 0.0)
    errors_ms = np.array([
        (beat_times[pi]-reference[ri])*1000
        for ri, pi in zip(sorted(matched_ref), sorted(matched_pred))
    ])
    result = {
        "n_reference":   n_ref,   "n_predicted":   n_pred,
        "n_matched":     n_matched,
        "n_missed":      n_ref  - n_matched,
        "n_false_alarm": n_pred - n_matched,
        "precision":     round(precision,  3),
        "recall":        round(recall,     3),
        "f_measure":     round(f_measure,  3),
        "tolerance_ms":  tolerance_ms,
        "mean_error_ms": round(float(errors_ms.mean()), 1) if len(errors_ms) else None,
        "std_error_ms":  round(float(errors_ms.std()),  1) if len(errors_ms) else None,
        "audio_start":   round(audio_start, 4),
    }
    print("\n" + "="*50)
    print("  madmom 박자 검출 정확도 (vs Beat Grid)")
    print("="*50)
    print(f"  모드                  : "
          f"{'online (실시간)' if REALTIME_MODE else 'offline (파일)'}")
    print(f"  audio_start (보정 후) : {audio_start:.3f}s")
    print(f"  허용 오차             : +-{tolerance_ms}ms")
    print(f"  정답 beat 수          : {n_ref}개")
    print(f"  검출 beat 수          : {n_pred}개")
    print(f"  매칭 성공             : {n_matched}개")
    print(f"  놓친 beat             : {n_ref-n_matched}개  (Recall 손실)")
    print(f"  오탐 beat             : {n_pred-n_matched}개  (Precision 손실)")
    print(f"  Precision             : {precision:.3f}")
    print(f"  Recall                : {recall:.3f}")
    print(f"  F-measure             : {f_measure:.3f}  <- 핵심 지표")
    if len(errors_ms):
        print(f"  평균 오차             : {errors_ms.mean():+.1f}ms")
        print(f"  표준편차              : {errors_ms.std():.1f}ms")
    print("="*50)
    return result
