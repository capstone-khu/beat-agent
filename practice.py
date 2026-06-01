import sys
print("실행 python:", sys.executable)

# ── Python 3.10+ / madmom 호환성 패치 ──────────────────────────
import collections
import collections.abc
if sys.version_info >= (3, 10):
    for _name in (
        "Callable", "Iterable", "Iterator", "Generator",
        "Mapping", "MutableMapping", "MutableSequence",
        "Sequence", "Set", "MutableSet",
    ):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))

# ── NumPy 1.24+ deprecated alias 패치 ───────────────────────────
import numpy as _np
for _alias, _builtin in (
    ("float", float), ("int", int), ("complex", complex),
    ("bool", bool), ("object", object), ("str", str),
):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _builtin)
del _np, _alias, _builtin

import numpy as np
import librosa
import pretty_midi
from scipy.spatial.distance import cdist
import subprocess, os, warnings, json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from BeatNetPlus.inference import BeatNetPlusInference

warnings.filterwarnings("ignore", category=FutureWarning)


def _set_korean_font():
    candidates = ["Malgun Gothic","AppleGothic","NanumGothic",
                  "NanumBarunGothic","Nanum Gothic","DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rc("font", family=name)
            print(f"[폰트] {name} 적용")
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_set_korean_font()

BEATNET_WEIGHTS   = "BeatNet-Plus/src/BeatNetPlus/models/af_non_percussive_weights.pt"
TOLERANCE_MS      = 70
SUBDIVISION_RATIO = 0.70
MAX_GAP_RATIO     = 1.6


def run_beatnet(audio_path: str) -> np.ndarray:
    estimator = BeatNetPlusInference(
        BEATNET_WEIGHTS, mode="online", inference_model="PF", device="cpu")
    output = estimator.process(audio_path)
    print(f"[BeatNet+] 검출 beat: {len(output)}개 "
          f"(downbeat: {int((output[:,1]==1).sum())}개)  "
          f"마지막 beat: {output[-1,0]:.3f}s")
    return output


def build_beat_grid_from_midi(midi_path: str) -> np.ndarray:
    midi       = pretty_midi.PrettyMIDI(midi_path)
    beat_times = midi.get_beats()
    print(f"[Beat Grid] {len(beat_times)}개  "
          f"평균 간격: {np.diff(beat_times).mean()*1000:.1f}ms")
    return beat_times


# ══════════════════════════════════════════════════════════════
#  Step 1: subdivision beat 제거
# ══════════════════════════════════════════════════════════════
def remove_subdivision_beats(
    beat_times, beat_grid, audio_start, beat_interval,
    ratio=SUBDIVISION_RATIO,
) -> np.ndarray:
    threshold = beat_interval * ratio
    ref       = beat_grid + audio_start
    beats     = list(beat_times)
    i, removed = 0, 0

    while i < len(beats) - 1:
        gap = beats[i+1] - beats[i]
        if gap < threshold:
            dist_i  = min(abs(ref - beats[i]))
            dist_i1 = min(abs(ref - beats[i+1]))
            if dist_i <= dist_i1:
                beats.pop(i+1)
            else:
                beats.pop(i)
            removed += 1
        else:
            i += 1

    result = np.array(beats)
    print(f"[Subdivision 제거] {removed}개 제거 "
          f"({len(beat_times)}개 → {len(result)}개)  "
          f"평균 간격: {np.diff(result).mean()*1000:.1f}ms")
    return result


# ══════════════════════════════════════════════════════════════
#  Step 2: Coarse align (median 기반)
#
#  기존: 첫 beat만 보고 정수박 offset 결정
#        → 첫 beat가 약간 어긋나면 전체가 밀림
#
#  개선: 전체 beat와 Beat Grid를 1:1 nearest 매핑 후
#        offset 분포의 median을 정수박 단위로 반올림
#        → 첫 beat 오류에 영향 받지 않음
# ══════════════════════════════════════════════════════════════
def coarse_align(
    beat_grid: np.ndarray,
    beat_times: np.ndarray,
    audio_start: float,
    beat_interval: float,
) -> float:
    """
    전체 beat의 median offset으로 정수박 단위 coarse 보정.
    반환: 보정된 audio_start
    """
    ref = beat_grid + audio_start

    # 각 BeatNet+ beat에 대해 가장 가까운 grid beat와의 차이 계산
    raw_offsets = []
    for b in beat_times:
        diffs = np.abs(ref - b)
        ri    = int(np.argmin(diffs))
        raw_offsets.append(b - ref[ri])

    # 박자 주기로 fold (모두 -beat_interval/2 ~ +beat_interval/2 범위로)
    folded = [(o + beat_interval/2) % beat_interval - beat_interval/2
              for o in raw_offsets]
    median_offset  = float(np.median(folded))

    # 정수박 단위로 반올림한 coarse offset
    n_beats_offset = round((beat_times[0] - (ref[0] + median_offset))
                            / beat_interval)
    coarse_offset  = median_offset + n_beats_offset * beat_interval
    new_start      = audio_start + coarse_offset

    print(f"[Coarse align] median fold offset: {median_offset*1000:+.1f}ms  "
          f"→ {n_beats_offset}박 보정  "
          f"audio_start: {audio_start:.3f}s → {new_start:.3f}s")
    return new_start


# ══════════════════════════════════════════════════════════════
#  Step 3: Beat Grid snap 보간
#
#  기존: 보간을 align 전에 수행 → 보간된 beat가 Grid와 안 맞음
#
#  개선: coarse align 후 보간 수행
#        누락 구간의 beat를 등간격이 아닌
#        aligned Beat Grid 위치로 snap해서 삽입
#        → 보간된 beat가 정답 위치에 정확히 놓임
# ══════════════════════════════════════════════════════════════
def fill_missing_beats_snapped(
    beat_times:    np.ndarray,
    beat_grid:     np.ndarray,
    audio_start:   float,
    beat_interval: float,
    max_gap_ratio: float = MAX_GAP_RATIO,
) -> np.ndarray:
    """
    누락 beat를 aligned Beat Grid 위치로 snap해서 삽입.
    coarse align 이후에 호출해야 한다.
    """
    threshold   = beat_interval * max_gap_ratio
    ref         = beat_grid + audio_start   # coarse align된 grid
    result      = list(beat_times)
    inserted    = 0
    i           = 0

    while i < len(result) - 1:
        gap = result[i+1] - result[i]
        if gap > threshold:
            n_miss = round(gap / beat_interval) - 1
            if n_miss > 0:
                # 누락 구간 중간 시각들에 가장 가까운 grid 위치 찾기
                for k in range(1, n_miss + 1):
                    approx_t = result[i] + k * (gap / (n_miss + 1))
                    diffs    = np.abs(ref - approx_t)
                    snap_t   = float(ref[int(np.argmin(diffs))])
                    # 이미 있는 beat와 너무 가까우면 skip
                    too_close = any(
                        abs(snap_t - existing) < beat_interval * 0.3
                        for existing in result
                    )
                    if not too_close:
                        result.insert(i + k, snap_t)
                        inserted += 1
        i += 1

    result = np.array(sorted(set(result)))
    print(f"[Snap 보간] {inserted}개 삽입 "
          f"({len(beat_times)}개 → {len(result)}개)  "
          f"평균 간격: {np.diff(result).mean()*1000:.1f}ms")
    return result


# ══════════════════════════════════════════════════════════════
#  Step 4: Fine align (DTW median)
# ══════════════════════════════════════════════════════════════
def fine_align(
    beat_grid:     np.ndarray,
    beat_times:    np.ndarray,
    audio_start:   float,
    beat_interval: float,
) -> tuple:
    """
    DTW 1:1 매핑의 median offset으로 ms 단위 미세 보정.
    반환: (보정된 audio_start, beat_times)
    """
    ref = beat_grid + audio_start

    if len(beat_times) < 2 or len(ref) < 2:
        return audio_start, beat_times

    D, wp = librosa.sequence.dtw(
        C=cdist(ref.reshape(-1,1), beat_times.reshape(-1,1))
    )
    wp = np.array(wp[::-1])

    seen_ref   = {}
    seen_query = {}
    for r_idx, q_idx in wp:
        seen_ref.setdefault(r_idx, []).append(q_idx)
        seen_query.setdefault(q_idx, []).append(r_idx)

    offsets = []
    for r_idx, q_indices in seen_ref.items():
        if len(q_indices) == 1:
            q_idx = q_indices[0]
            if len(seen_query[q_idx]) == 1:
                diff = beat_times[q_idx] - ref[r_idx]
                if abs(diff) < beat_interval * 0.5:
                    offsets.append(diff)

    if not offsets:
        print("[Fine align] 유효 매핑 없음 -> 유지")
        return audio_start, beat_times

    fine_offset = float(np.median(offsets))
    new_start   = audio_start + fine_offset
    print(f"[Fine align] 매핑 {len(offsets)}개 -> "
          f"offset {fine_offset*1000:+.1f}ms  "
          f"audio_start: {audio_start:.3f}s → {new_start:.3f}s")
    print(f"[Fine align] 보정 후 Grid 첫 박: {beat_grid[0]+new_start:.3f}s  "
          f"BeatNet+ 첫 beat: {beat_times[0]:.3f}s")
    return new_start, beat_times


def evaluate_beatnet_detection(
    beat_grid, beat_times, audio_start, tolerance_ms=TOLERANCE_MS,
) -> dict:
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
        "mean_error_ms": round(float(errors_ms.mean()),1) if len(errors_ms) else None,
        "std_error_ms":  round(float(errors_ms.std()), 1) if len(errors_ms) else None,
        "audio_start":   round(audio_start, 4),
    }

    print("\n" + "="*50)
    print("  BeatNet+ 박자 검출 정확도 (vs Beat Grid)")
    print("="*50)
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


class ViolinRhythmAgent:
    """
    바이올린 연주 박자 정확도 평가 에이전트 (실시간 전용).

    처리 파이프라인
    ──────────────────────────────────────────────────────────
    BeatNet+ 원본
      1. subdivision 제거  (간격 < 기대값×70%)
      2. Coarse align      (전체 beat median 기반, 박자 fold)
      3. Snap 보간         (누락 beat를 Grid 위치로 snap 삽입)
      4. Fine align        (DTW median ms 단위)
      5. 타이밍 점수 산출
    """

    def __init__(self, midi_path: str, chunk_duration: float = 3.0):
        self.chunk_duration = chunk_duration
        self.midi_path      = midi_path
        self._prev_timing   = None

        midi               = pretty_midi.PrettyMIDI(midi_path)
        self.bpm           = midi.get_tempo_changes()[1][0]
        self.beat_interval = 60.0 / self.bpm

        self.midi_notes    = self._load_midi_notes(midi_path)
        self.beat_grid     = build_beat_grid_from_midi(midi_path)
        print(f"[Beat Grid] 첫 박: {self.beat_grid[0]:.3f}s  "
              f"마지막 박: {self.beat_grid[-1]:.3f}s  "
              f"BPM: {self.bpm:.1f}  간격: {self.beat_interval*1000:.0f}ms")

    def _load_midi_notes(self, midi_path):
        midi      = pretty_midi.PrettyMIDI(midi_path)
        all_notes = []
        for inst in midi.instruments:
            for n in inst.notes:
                all_notes.append({"onset": n.start, "pitch": n.pitch})
        all_notes.sort(key=lambda x: x["onset"])
        melody, i = [], 0
        while i < len(all_notes):
            cluster = [all_notes[i]]
            j = i + 1
            while j < len(all_notes) and \
                  all_notes[j]["onset"] - all_notes[i]["onset"] < 0.03:
                cluster.append(all_notes[j])
                j += 1
            melody.append(max(cluster, key=lambda x: x["pitch"]))
            i = j
        return melody

    def _to_wav(self, src_path):
        if src_path.lower().endswith(".wav"):
            return src_path
        wav_path = os.path.splitext(src_path)[0] + ".wav"
        if not os.path.exists(wav_path):
            subprocess.run([
                "ffmpeg", "-i", src_path,
                "-vn", "-acodec", "pcm_s16le", "-ac", "1", wav_path
            ], check=True, capture_output=True)
            print(f"[WAV] 변환 완료: {wav_path}")
        return wav_path

    def _detect_audio_start(self, y, sr, top_db=28):
        _, idx = librosa.effects.trim(y, top_db=top_db)
        t = idx[0] / sr
        print(f"[시작점] {t:.3f}s")
        return t

    def _score_timing(self, grid_onsets, beat_times):
        if len(grid_onsets) < 2 or len(beat_times) < 2:
            return 0.0, []
        D, wp  = librosa.sequence.dtw(
            C=cdist(grid_onsets.reshape(-1,1), beat_times.reshape(-1,1)))
        wp     = np.array(wp[::-1])
        signed = [float(beat_times[j]-grid_onsets[i]) for i,j in wp]
        errors = [abs(e) for e in signed]
        rel    = [e / self.beat_interval for e in errors]
        raw    = float(np.exp(-np.mean(rel) * 3))
        score  = raw if self._prev_timing is None \
                 else 0.7*self._prev_timing + 0.3*raw
        self._prev_timing = score
        return score, signed

    @staticmethod
    def _timing_label(s):
        if s >= 0.80: return "정확"
        if s >= 0.55: return "보통"
        return "불안정"

    @staticmethod
    def _drift_label(signed):
        if not signed: return "측정불가"
        ms = float(np.mean(signed)) * 1000
        if   ms >  80: return f"늦게 연주 +{ms:.0f}ms"
        elif ms < -80: return f"빠르게 연주 {ms:.0f}ms"
        else:          return f"박자 정확 ({ms:+.0f}ms)"

    def process(self, audio_path: str) -> tuple:
        print("[BeatNet+] beat 추적 중...")
        raw            = run_beatnet(audio_path)
        beat_times     = raw[:, 0]
        downbeat_times = raw[raw[:, 1] == 1, 0]
        print(f"[BeatNet+] 범위: {beat_times[0]:.3f}s ~ {beat_times[-1]:.3f}s  "
              f"평균 간격: {np.diff(beat_times).mean()*1000:.1f}ms")

        wav_path    = self._to_wav(audio_path)
        y, sr       = librosa.load(wav_path, sr=None, mono=True)
        print(f"[오디오] 길이: {len(y)/sr:.2f}s  sr: {sr}")
        audio_start = self._detect_audio_start(y, sr)

        beat_times     = beat_times[beat_times >= audio_start]
        downbeat_times = downbeat_times[downbeat_times >= audio_start]

        # 1. subdivision 제거
        beat_times = remove_subdivision_beats(
            beat_times    = beat_times,
            beat_grid     = self.beat_grid,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )

        # 2. Coarse align (median 기반)
        audio_start = coarse_align(
            beat_grid     = self.beat_grid,
            beat_times    = beat_times,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )

        # 3. Snap 보간 (coarse align 후 Grid 위치로 snap)
        beat_times = fill_missing_beats_snapped(
            beat_times    = beat_times,
            beat_grid     = self.beat_grid,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )

        # 4. Fine align (DTW median)
        audio_start, beat_times = fine_align(
            beat_grid     = self.beat_grid,
            beat_times    = beat_times,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )
        shifted_grid = self.beat_grid + audio_start

        total_chunks = max(1, int(
            (beat_times[-1] - audio_start) / self.chunk_duration))
        results      = []
        print("\n===== 바이올린 박자 평가 시작 =====")

        for i in range(total_chunks):
            t_start = audio_start + i * self.chunk_duration
            t_end   = audio_start + (i+1) * self.chunk_duration

            grid_seg   = shifted_grid[
                (shifted_grid >= t_start) & (shifted_grid < t_end)]
            beats_seg  = beat_times[
                (beat_times  >= t_start) & (beat_times  < t_end)]
            dbeats_seg = downbeat_times[
                (downbeat_times >= t_start) & (downbeat_times < t_end)]

            print(f"\n[{t_start:.1f}s ~ {t_end:.1f}s]  "
                  f"Beat Grid: {len(grid_seg)}박  "
                  f"BeatNet+: {len(beats_seg)}beat "
                  f"(downbeat {len(dbeats_seg)}개)")

            timing_score, signed = self._score_timing(grid_seg, beats_seg)
            t_label = self._timing_label(timing_score)
            d_label = self._drift_label(signed)
            print(f"  타이밍: {timing_score:.3f}  {t_label}")
            print(f"  밀림:   {d_label}")

            results.append({
                "start_time":       round(t_start, 2),
                "end_time":         round(t_end, 2),
                "grid_count":       int(len(grid_seg)),
                "beat_count":       int(len(beats_seg)),
                "downbeat_count":   int(len(dbeats_seg)),
                "timing_score":     round(timing_score, 3),
                "timing_label":     t_label,
                "drift_label":      d_label,
                "signed_errors_ms": [round(e*1000,1) for e in signed],
            })

        return (json.dumps(results, ensure_ascii=False, indent=2),
                audio_start, beat_times, downbeat_times, self.beat_grid)


def evaluate_performance(json_results: str) -> dict:
    data = json.loads(json_results)
    if not data:
        return {"error": "결과 없음"}
    timing_avg = float(np.mean([c["timing_score"] for c in data]))
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


def plot_beat_comparison(
    beat_grid, beat_times, downbeat_times=None,
    audio_start=0.0, detection_stats=None,
    title="Beat Grid vs BeatNet+",
):
    shifted_grid = beat_grid + audio_start

    errors_ms, matched_grid, matched_beat = [], [], []
    for g in shifted_grid:
        diffs = np.abs(beat_times - g)
        idx   = int(np.argmin(diffs))
        if diffs[idx] < 0.5:
            errors_ms.append((beat_times[idx]-g)*1000)
            matched_grid.append(g)
            matched_beat.append(beat_times[idx])
    errors_ms    = np.array(errors_ms)
    matched_grid = np.array(matched_grid)
    matched_beat = np.array(matched_beat)

    colors = {"grid":"#4A90D9","beat":"#E05C5C","downbeat":"#C0392B",
              "match":"#2ECC71","error_pos":"#E67E22","error_neg":"#8E44AD"}

    n_rows   = 4 if detection_stats else 3
    h_ratios = [2,2,3,2] if detection_stats else [2,2,3]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4*n_rows),
                             gridspec_kw={"height_ratios": h_ratios})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    x_min = min(shifted_grid[0], beat_times[0]) - 0.2
    x_max = max(shifted_grid[-1], beat_times[-1]) + 0.2

    ax1 = axes[0]
    ax1.set_title("beat 위치 타임라인", fontsize=11, pad=6)
    ax1.vlines(shifted_grid, 0.6, 1.4, color=colors["grid"],
               linewidth=1.5, alpha=0.8, label="Beat Grid")
    ax1.scatter(shifted_grid, np.ones(len(shifted_grid)),
                color=colors["grid"], s=40, zorder=3)
    ax1.vlines(beat_times, -0.4, 0.4, color=colors["beat"],
               linewidth=1.5, alpha=0.8, label="BeatNet+ (보정 후)")
    ax1.scatter(beat_times, np.zeros(len(beat_times)),
                color=colors["beat"], s=40, zorder=3)
    if downbeat_times is not None and len(downbeat_times) > 0:
        valid_db = downbeat_times[downbeat_times >= audio_start]
        if len(valid_db) > 0:
            ax1.scatter(valid_db, np.zeros(len(valid_db)),
                        color=colors["downbeat"], s=110, marker="D",
                        zorder=4, label="Downbeat")
    for g, b in zip(matched_grid, matched_beat):
        ax1.plot([g,b],[1,0], color=colors["match"],
                 linewidth=0.8, alpha=0.5, linestyle="--")
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(["BeatNet+","Beat Grid"], fontsize=10)
    ax1.set_xlabel("시간 (초)", fontsize=10)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(-0.8, 1.8)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="x", linestyle=":", alpha=0.4)
    ax1.spines[["top","right","left"]].set_visible(False)

    ax2 = axes[1]
    ax2.set_title("박자별 오차  (+ = 늦음,  - = 빠름)", fontsize=11, pad=6)
    if len(errors_ms) > 0:
        bar_colors = [colors["error_pos"] if e>=0 else colors["error_neg"]
                      for e in errors_ms]
        ax2.bar(matched_grid, errors_ms, width=0.04,
                color=bar_colors, alpha=0.85, zorder=3)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.axhline( TOLERANCE_MS, color=colors["error_pos"],
                    linewidth=0.9, linestyle="--", alpha=0.7,
                    label=f"+{TOLERANCE_MS}ms")
        ax2.axhline(-TOLERANCE_MS, color=colors["error_neg"],
                    linewidth=0.9, linestyle="--", alpha=0.7,
                    label=f"-{TOLERANCE_MS}ms")
        ax2.axhline(errors_ms.mean(), color="gray", linewidth=1.2,
                    linestyle="-.", alpha=0.9,
                    label=f"평균 {errors_ms.mean():+.1f}ms")
    ax2.set_xlabel("시간 (초)", fontsize=10)
    ax2.set_ylabel("오차 (ms)", fontsize=10)
    ax2.set_xlim(x_min, x_max)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="x", linestyle=":", alpha=0.4)
    ax2.spines[["top","right"]].set_visible(False)

    ax3 = axes[2]
    ax3.set_title("오차 분포 히스토그램", fontsize=11, pad=6)
    if len(errors_ms) > 0:
        bins = np.arange(-300, 301, 20)
        n, b, patches = ax3.hist(errors_ms, bins=bins,
                                  edgecolor="white", linewidth=0.5, zorder=3)
        for patch, left in zip(patches, b[:-1]):
            patch.set_facecolor(colors["error_pos"] if left>=0
                                else colors["error_neg"])
            patch.set_alpha(0.8)
        ax3.axvline(0,             color="black",          linewidth=1.0)
        ax3.axvline( TOLERANCE_MS, color=colors["error_pos"],
                    linewidth=0.9, linestyle="--", alpha=0.7)
        ax3.axvline(-TOLERANCE_MS, color=colors["error_neg"],
                    linewidth=0.9, linestyle="--", alpha=0.7)
        ax3.axvline(errors_ms.mean(), color="gray", linewidth=1.2,
                    linestyle="-.", alpha=0.9,
                    label=f"평균 {errors_ms.mean():+.1f}ms")
        within_tol = (np.abs(errors_ms)<=TOLERANCE_MS).mean()*100
        within_100 = (np.abs(errors_ms)<=100).mean()*100
        ax3.text(0.98, 0.95,
                 f"+-{TOLERANCE_MS}ms 이내  {within_tol:.0f}%\n"
                 f"+-100ms 이내  {within_100:.0f}%\n"
                 f"표준편차      {errors_ms.std():.1f}ms",
                 transform=ax3.transAxes, fontsize=9, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor="white", edgecolor="#ccc", alpha=0.9))
    ax3.set_xlabel("오차 (ms)", fontsize=10)
    ax3.set_ylabel("beat 수", fontsize=10)
    ax3.legend(loc="upper left", fontsize=9)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)
    ax3.spines[["top","right"]].set_visible(False)

    if detection_stats and n_rows == 4:
        ax4 = axes[3]
        ax4.set_title("BeatNet+ 검출 정확도", fontsize=11, pad=6)
        ax4.axis("off")
        metrics = [
            ("Precision", detection_stats["precision"],
             "검출된 beat 중 정답 비율\n(낮으면 없는 박자를 만들어냄)"),
            ("Recall",    detection_stats["recall"],
             "정답 beat 중 검출 비율\n(낮으면 박자를 놓침)"),
            ("F-measure", detection_stats["f_measure"],
             "Precision x Recall 조화평균\n(전체 검출 정확도)"),
        ]
        for k, (name, val, desc) in enumerate(metrics):
            x     = 0.05 + k*0.32
            color = ("#2ECC71" if val>=0.80
                     else "#E67E22" if val>=0.55 else "#E05C5C")
            ax4.add_patch(plt.Rectangle((x, 0.15), 0.28, 0.70,
                transform=ax4.transAxes, facecolor=color,
                alpha=0.15, edgecolor=color, linewidth=1.5))
            ax4.text(x+0.14, 0.72, name, transform=ax4.transAxes,
                     fontsize=11, fontweight="bold", ha="center", color=color)
            ax4.text(x+0.14, 0.50, f"{val:.3f}", transform=ax4.transAxes,
                     fontsize=22, fontweight="bold", ha="center", color=color)
            ax4.text(x+0.14, 0.25, desc, transform=ax4.transAxes,
                     fontsize=8, ha="center", color="gray", linespacing=1.5)
        info = (f"정답 {detection_stats['n_reference']}개  |  "
                f"검출 {detection_stats['n_predicted']}개  |  "
                f"매칭 {detection_stats['n_matched']}개  |  "
                f"놓침 {detection_stats['n_missed']}개  |  "
                f"오탐 {detection_stats['n_false_alarm']}개  |  "
                f"audio_start {detection_stats['audio_start']:.3f}s")
        ax4.text(0.5, 0.06, info, transform=ax4.transAxes,
                 fontsize=8, ha="center", color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = "beat_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[시각화] {out_path} 저장 완료")


if __name__ == "__main__":
    MIDI_PATH  = "twinkle.mid"
    AUDIO_PATH = "performance.mp4"

    agent = ViolinRhythmAgent(MIDI_PATH, chunk_duration=3.0)

    res, aligned_audio_start, beat_times_final, \
        downbeat_times_final, beat_grid_final = agent.process(AUDIO_PATH)

    gaps = np.diff(beat_times_final)
    print(f"\n[진단] 보정 후 beat 간격:")
    print(f"  평균: {gaps.mean()*1000:.1f}ms  "
          f"표준편차: {gaps.std()*1000:.1f}ms  "
          f"기대값: {agent.beat_interval*1000:.1f}ms")
    print(f"  Beat Grid {len(beat_grid_final)}개 / "
          f"BeatNet+ {len(beat_times_final)}개")

    print(f"\n[진단] 정렬 후 첫 5개 비교:")
    print(f"  Beat Grid: "
          f"{np.round(beat_grid_final[:5] + aligned_audio_start, 3)}")
    print(f"  BeatNet+ : {np.round(beat_times_final[:5], 3)}")
    print(f"  첫 beat 차이: "
          f"{(beat_times_final[0]-(beat_grid_final[0]+aligned_audio_start))*1000:.1f}ms")

    detection_stats = evaluate_beatnet_detection(
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
        downbeat_times  = downbeat_times_final,
        audio_start     = aligned_audio_start,
        detection_stats = detection_stats,
        title           = "Beat Grid vs BeatNet+ (median coarse + snap 보간)",
    )
