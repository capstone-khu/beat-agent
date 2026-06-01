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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import madmom
import madmom.features.beats as mf_beats

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

TOLERANCE_MS  = 70
REALTIME_MODE = True


# ══════════════════════════════════════════════════════════════
#  박자 State 정의 (이미지 기준)
#  R-S0 GOOD  : drift ±80ms 이내 + score >= 0.80
#  R-S1 EARLY : drift < -80ms  (박자보다 일찍 연주)
#  R-S2 LATE  : drift > +80ms  (박자보다 늦게 연주)
#  R-S3 FAST  : beat 과다 (빠른 템포)
#  R-S4 SLOW  : beat 부족 or score 낮음 (느린 템포)
#
#  Action 정의 (이미지 기준)
#  SA-06 RHYTHM_WAIT       : R-S1 EARLY
#  SA-07 RHYTHM_CATCH_UP   : R-S2 LATE
#  SA-08 TEMPO_SLOW_DOWN   : R-S3 FAST
#  SA-09 TEMPO_SPEED_UP    : R-S4 SLOW
#  SA-10 POSITIVE_RHYTHM   : R-S0 GOOD
#  SA-11 CALL_SUPERVISOR   : 연속 실패 3회 이상 or 나머지 두 영역 모두 GOOD
# ══════════════════════════════════════════════════════════════

# State ID → 문자열 매핑
STATES  = ["GOOD", "EARLY", "LATE", "FAST", "SLOW"]

# Action ID → 문자열 매핑
ACTIONS = [
    "POSITIVE_RHYTHM",   # SA-10
    "RHYTHM_WAIT",       # SA-06
    "RHYTHM_CATCH_UP",   # SA-07
    "TEMPO_SLOW_DOWN",   # SA-08
    "TEMPO_SPEED_UP",    # SA-09
    "CALL_SUPERVISOR",   # SA-11
]

# Action → Action ID 매핑 (이미지 기준)
ACTION_ID = {
    "POSITIVE_RHYTHM":  "SA-10",
    "RHYTHM_WAIT":      "SA-06",
    "RHYTHM_CATCH_UP":  "SA-07",
    "TEMPO_SLOW_DOWN":  "SA-08",
    "TEMPO_SPEED_UP":   "SA-09",
    "CALL_SUPERVISOR":  "SA-11",
}

# Action별 피드백 메시지 (이미지 기준)
ACTION_FEEDBACK = {
    "POSITIVE_RHYTHM":  "잘 하고 있습니다. 계속 유지하세요",
    "RHYTHM_WAIT":      "박자보다 일찍 연주하고 있습니다. 박자를 맞추세요",
    "RHYTHM_CATCH_UP":  "박자보다 늦게 연주하고 있습니다. 박자를 맞추세요",
    "TEMPO_SLOW_DOWN":  "템포가 빠릅니다. 속도를 늦추세요",
    "TEMPO_SPEED_UP":   "템포가 느립니다. 속도를 높이세요",
    "CALL_SUPERVISOR":  "여기서 계속 같은 문제가 발생해요. 나중에 반복 연습하면서 개선해봐요",
}

# 룰베이스 기본 액션 (Q테이블 학습 데이터 없을 때)
DEFAULT_ACTION = {
    "GOOD":  "POSITIVE_RHYTHM",
    "EARLY": "RHYTHM_WAIT",
    "LATE":  "RHYTHM_CATCH_UP",
    "FAST":  "TEMPO_SLOW_DOWN",
    "SLOW":  "TEMPO_SPEED_UP",
}

# Reward 정의 (이미지 기준)
REWARD_TABLE = {
    "good_transition":        +1.0,   # 액션 후 State GOOD 전환
    "partial_improvement":    +0.5,   # 심각 → 경미로 개선 (예: FAST→SLOW)
    "no_change":              -0.3,   # 액션 후 State 변화 없음
    "supervisor_effective":   +0.8,   # CALL_SUPERVISOR 후 개선 있음
    "supervisor_ineffective": -0.5,   # CALL_SUPERVISOR 후 개선 없음
    "deterioration":          -0.8,   # 악화됨 (GOOD → BAD)
}

# Q-learning 하이퍼파라미터
ALPHA = 0.1   # 학습률
GAMMA = 0.9   # 할인율


# ══════════════════════════════════════════════════════════════
#  Q테이블 클래스 — 에이전트가 개별 소유
# ══════════════════════════════════════════════════════════════
class RhythmQTable:
    """
    박자 에이전트 전용 Q테이블.

    구조: Q[state][action] = float (초기값 0.0)
    업데이트: Q(S,A) ← Q(S,A) + α[R + γ·maxQ(S',A') - Q(S,A)]
    """

    def __init__(self, states: list, actions: list,
                 alpha: float = ALPHA, gamma: float = GAMMA):
        self.states  = states
        self.actions = actions
        self.alpha   = alpha
        self.gamma   = gamma
        # Q테이블 초기화 (모든 값 0.0)
        self.table: dict[str, dict[str, float]] = {
            s: {a: 0.0 for a in actions} for s in states
        }

    def best_action(self, state: str) -> str:
        """현재 State에서 Q값이 가장 높은 Action 반환."""
        q_row = self.table[state]
        # 모든 Q값이 0이면 (초기 상태) 룰베이스 기본 액션 반환
        if all(v == 0.0 for v in q_row.values()):
            return DEFAULT_ACTION.get(state, "POSITIVE_RHYTHM")
        return max(q_row, key=lambda a: q_row[a])

    def update(self, state: str, action: str,
               reward: float, next_state: str) -> float:
        """
        Q-learning 업데이트.
        Q(S,A) ← Q(S,A) + α[R + γ·maxQ(S',A') - Q(S,A)]

        반환: 업데이트된 Q(S,A) 값
        """
        q_current  = self.table[state][action]
        max_q_next = max(self.table[next_state].values())
        td_target  = reward + self.gamma * max_q_next
        q_new      = q_current + self.alpha * (td_target - q_current)
        self.table[state][action] = q_new
        return q_new

    def get(self, state: str, action: str) -> float:
        return self.table[state][action]

    def summary(self) -> str:
        lines = ["[Q-Table 현황]"]
        for s in self.states:
            row = "  ".join(
                f"{a}:{v:+.3f}" for a, v in self.table[s].items()
            )
            lines.append(f"  {s:6s} | {row}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Reward 산출 함수
# ══════════════════════════════════════════════════════════════

# State 심각도 순위 (숫자가 클수록 나쁨)
_STATE_SEVERITY = {"GOOD": 0, "EARLY": 2, "LATE": 2, "FAST": 1, "SLOW": 1}

def compute_reward(prev_state: str, curr_state: str, action: str) -> float:
    """
    이전 State, 현재 State, 선택한 Action으로 Reward 산출.

    규칙 (이미지 Reward/Penalty 테이블 기준)
    ─────────────────────────────────────────
    +1.0  State가 GOOD으로 전환
    +0.5  심각(EARLY/LATE) → 경미(FAST/SLOW)로 개선
    +0.8  CALL_SUPERVISOR 후 현재 State가 이전보다 나아짐
    -0.5  CALL_SUPERVISOR 후 개선 없음
    -0.3  State 변화 없음 (제자리)
    -0.8  GOOD → 나쁜 State로 악화
    """
    if curr_state == "GOOD":
        if prev_state == "GOOD":
            return REWARD_TABLE["no_change"]     # 유지 (변화 없음)
        return REWARD_TABLE["good_transition"]   # GOOD 전환

    if prev_state == "GOOD" and curr_state != "GOOD":
        return REWARD_TABLE["deterioration"]     # 악화

    if action == "CALL_SUPERVISOR":
        prev_sev = _STATE_SEVERITY.get(prev_state, 2)
        curr_sev = _STATE_SEVERITY.get(curr_state, 2)
        if curr_sev < prev_sev:
            return REWARD_TABLE["supervisor_effective"]
        return REWARD_TABLE["supervisor_ineffective"]

    # 심각(EARLY/LATE) → 경미(FAST/SLOW): 부분 개선
    severe  = {"EARLY", "LATE"}
    mild    = {"FAST", "SLOW"}
    if prev_state in severe and curr_state in mild:
        return REWARD_TABLE["partial_improvement"]

    # State 변화 없음
    if prev_state == curr_state:
        return REWARD_TABLE["no_change"]

    # 그 외 (나쁜 State 간 이동 등) — 변화 없음으로 처리
    return REWARD_TABLE["no_change"]


# ══════════════════════════════════════════════════════════════
#  madmom / Beat Grid 유틸리티 (변경 없음)
# ══════════════════════════════════════════════════════════════

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
    resampled  = np.array([first_beat + i * beat_interval
                           for i in range(n_beats)])
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
        "mean_error_ms": round(float(errors_ms.mean()),1) if len(errors_ms) else None,
        "std_error_ms":  round(float(errors_ms.std()), 1) if len(errors_ms) else None,
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


# ══════════════════════════════════════════════════════════════
#  박자 State 판별
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
#  슈퍼바이저 전달
# ══════════════════════════════════════════════════════════════
def report_to_supervisor(
    supervisor,
    action:      str,
    curr_state:  str,
    reward:      float | None,
    q_value:     float,
    measure:     int,
    fail_count:  int | None = None,   # 미사용 (하위 호환 유지용)
    meta:        dict | None = None,
) -> dict:
    """
    슈퍼바이저에 표준 페이로드를 전달한다.
    supervisor가 None이면 로컬 출력만 수행.

    페이로드 스펙
    ─────────────────────────────────────────────────────
    {
      "agent":     "rhythm",
      "measure":   3,
      "state":     "LATE",
      "action_id": "SA-07",
      "action":    "RHYTHM_CATCH_UP",
      "feedback":  "박자보다 늦게 ...",
      "reward":    -0.3,          # 직전 액션 대비 평가 (첫 번째는 null)
      "q":         0.0700,        # 갱신 후 Q[state][action]
      "meta": {
        "index":       4,         # 0-based chunk 순번
        "half":        1,         # 1=전반, 2=후반
        "start_time":  2.532,
        "end_time":    3.165,
        "score":       0.61,
        "drift_label": "LATE +92ms"
      }
    }
    """
    payload = {
        "agent":     "rhythm",
        "measure":   measure,
        "state":     curr_state,
        "action_id": ACTION_ID.get(action, "SA-10"),
        "action":    action,
        "feedback":  ACTION_FEEDBACK.get(action, ""),
        "reward":    reward,
        "q":         round(q_value, 4),
        "meta":      meta if meta is not None else {},
    }

    if supervisor is not None:
        # supervisor.receive(payload)  # 실제 연결 시 활성화
        pass

    print(f"[Supervisor] {json.dumps(payload, ensure_ascii=False)}")
    return payload


# ══════════════════════════════════════════════════════════════
#  ViolinRhythmAgent
# ══════════════════════════════════════════════════════════════
class ViolinRhythmAgent:
    """
    바이올린 연주 박자 정확도 평가 에이전트.

    ── Q테이블 소유 ──
    각 에이전트는 자신의 Q테이블을 보유하며,
    Q값을 조회해 최적 액션을 선택한다.
    초기에는 학습 데이터가 없으므로 룰베이스 기본 피드백을 사용.
    피드백 후 Reward를 산출해 Q테이블을 갱신하고 슈퍼바이저에 전달.

    ── Q-learning 업데이트 ──
    Q(S,A) ← Q(S,A) + α[R + γ·maxQ(S',A') - Q(S,A)]
    α=0.1, γ=0.9

    ── 처리 파이프라인 ──
    madmom RNN + DBN (BPM +-5% 제한)
      1. First-beat align
      2. BPM 리샘플 (MIDI BPM 간격, 누적 오차 제거)
      3. 타이밍 점수 산출
      4. State 판별 → Q테이블 조회 → 액션 선택
      5. Reward 산출 → Q테이블 업데이트 → 슈퍼바이저 보고
    """

    def __init__(self, midi_path: str,
                 beats_per_measure: int = 4,
                 call_supervisor_q_threshold: float = -0.3):
        """
        파라미터
        ────────
        midi_path                   : MIDI 파일 경로
        beats_per_measure           : 박자 수 / 마디 (기본 4 → 4/4박자)
        call_supervisor_q_threshold : 이 값 이하의 Q값이면 CALL_SUPERVISOR 선택
                                      (fail_count 대신 Q값으로 위임 판단)
        """
        self.midi_path               = midi_path
        self.beats_per_measure       = beats_per_measure
        self.beats_per_half_measure  = beats_per_measure // 2   # 반 마디 = 2박
        self.call_supervisor_q_threshold = call_supervisor_q_threshold
        self._prev_timing            = None

        midi               = pretty_midi.PrettyMIDI(midi_path)
        self.bpm           = midi.get_tempo_changes()[1][0]
        self.beat_interval = 60.0 / self.bpm
        # 반 마디 길이 (초)
        self.half_measure_duration = self.beat_interval * self.beats_per_half_measure
        self.midi_notes    = self._load_midi_notes(midi_path)
        self.beat_grid     = build_beat_grid_from_midi(midi_path)

        # ── 에이전트 개별 Q테이블 ──
        self.q_table = RhythmQTable(STATES, ACTIONS, alpha=ALPHA, gamma=GAMMA)

        print(f"[Beat Grid] 첫 박: {self.beat_grid[0]:.3f}s  "
              f"마지막 박: {self.beat_grid[-1]:.3f}s  "
              f"BPM: {self.bpm:.1f}  간격: {self.beat_interval*1000:.0f}ms")
        print(f"[반 마디]   {self.beats_per_half_measure}박 "
              f"= {self.half_measure_duration*1000:.0f}ms")
        print(f"[Q-Table]  초기화 완료 — States: {STATES}, Actions: {ACTIONS}")
        print(f"[Supervisor 위임] Q값 ≤ {self.call_supervisor_q_threshold} 시 CALL_SUPERVISOR")

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
        """
        타이밍 점수 산출 (detrended nearest-match 방식).
        반환: (smoothed_score, raw_score, signed_errors)

        ── 설계 원칙 ──────────────────────────────────────────────
        · raw_beat_times(원본 madmom) 기준으로 점수 계산
        · 단, BPM 차이로 인한 누적 드리프트(tempo drift)는 제거
          → 구간 내 평균 오차(drift_avg)를 빼고 잔차만 평가
          → '박자 간격의 일관성'이 아닌 '박자 내 위치 정확도' 측정
        · 드리프트 자체는 drift_label로 별도 전달

        ── 처리 흐름 ──────────────────────────────────────────────
        1. 각 grid beat에 nearest raw beat 1:1 매핑
        2. signed_errors 계산
        3. mean_drift(평균 오차) 제거 → residuals
        4. residuals 기준으로 exp 점수 산출
        5. EMA smoothing (0.5/0.5)
        """
        if len(grid_onsets) == 0 or len(beat_times) == 0:
            prev = self._prev_timing if self._prev_timing is not None else 0.0
            return prev, 0.0, []

        # 1:1 nearest 매핑
        signed = []
        for g in grid_onsets:
            diffs = np.abs(beat_times - g)
            ri    = int(np.argmin(diffs))
            signed.append(float(beat_times[ri] - g))

        # 구간 평균 드리프트 제거 → 잔차
        mean_drift = float(np.mean(signed))
        residuals  = [s - mean_drift for s in signed]
        rel        = [abs(r) / self.beat_interval for r in residuals]
        raw        = float(np.exp(-np.mean(rel) * 3))

        smoothed = raw if self._prev_timing is None \
                   else 0.5 * self._prev_timing + 0.5 * raw
        self._prev_timing = smoothed
        return smoothed, raw, signed

    def _drift_signed(self, grid_onsets, raw_beat_times):
        """
        원본 madmom beat 기준 drift 계산 (EMA 없음, _prev_timing 불변).
        detrend 없이 실제 절대 오차를 반환 → drift_label 산출용.
        반환: signed_errors list[float] (단위: 초)
        """
        if len(grid_onsets) == 0 or len(raw_beat_times) == 0:
            return []
        signed = []
        for g in grid_onsets:
            diffs = np.abs(raw_beat_times - g)
            ri    = int(np.argmin(diffs))
            signed.append(float(raw_beat_times[ri] - g))
        return signed

    @staticmethod
    def _timing_label(s):
        if s >= 0.80: return "ACCURATE"
        if s >= 0.55: return "MODERATE"
        return "UNSTABLE"

    @staticmethod
    def _drift_label(signed):
        if not signed: return "UNKNOWN"
        ms = float(np.mean(signed)) * 1000
        if   ms >  80: return f"LATE +{ms:.0f}ms"
        elif ms < -80: return f"EARLY {ms:.0f}ms"
        else:          return f"ON_TIME ({ms:+.0f}ms)"

    def process(self, audio_path: str) -> tuple:
        wav_path = self._to_wav(audio_path)

        print(f"[madmom] beat 추적 중...")
        beat_times = run_madmom(wav_path, self.bpm)

        y, sr       = librosa.load(wav_path, sr=None, mono=True)
        print(f"[오디오] 길이: {len(y)/sr:.2f}s  sr: {sr}")
        audio_start = self._detect_audio_start(y, sr)

        beat_times = beat_times[beat_times >= audio_start]

        audio_start, beat_times = align_to_grid(
            beat_grid     = self.beat_grid,
            beat_times    = beat_times,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )

        raw_beat_times = beat_times.copy()
        beat_times = resample_to_midi_bpm(
            beat_times    = beat_times,
            beat_grid     = self.beat_grid,
            audio_start   = audio_start,
            beat_interval = self.beat_interval,
        )

        shifted_grid = self.beat_grid + audio_start

        # ── 반 마디 단위 chunk 경계 생성 ────────────────────────
        # beat_grid는 beat 단위이므로 beats_per_half_measure 간격으로 슬라이싱
        # 예: 4/4박자 → 2박마다 하나의 chunk
        half = self.beats_per_half_measure
        # beat_grid 인덱스 기준 반 마디 시작점들
        half_measure_starts = self.beat_grid[::half] + audio_start
        total_chunks = len(half_measure_starts)

        results = []
        print("\n===== 바이올린 박자 평가 시작 (반 마디 단위) =====\n")

        for i in range(total_chunks):
            t_start = half_measure_starts[i]
            t_end   = (half_measure_starts[i + 1]
                       if i + 1 < total_chunks
                       else beat_times[-1] + self.beat_interval)

            # 이 chunk가 몇 번째 마디의 전반(1)/후반(2)인지
            measure_number = i // 2 + 1          # 1-based 마디 번호
            half_in_measure = i % 2 + 1          # 1 = 전반, 2 = 후반

            grid_seg      = shifted_grid[
                (shifted_grid >= t_start) & (shifted_grid < t_end)]
            beats_seg     = beat_times[
                (beat_times  >= t_start) & (beat_times  < t_end)]
            raw_beats_seg = raw_beat_times[
                (raw_beat_times >= t_start) & (raw_beat_times < t_end)]

            # ── 점수: raw_beats vs grid (detrended) ───────────────
            # · raw_beats = 원본 madmom beat (실제 연주 위치)
            # · _score_timing 내부에서 BPM 드리프트(tempo offset)를 제거하고
            #   박자 내 위치 정확도(잔차)만 평가
            # · drift_label은 _drift_signed로 절대 오차 기준 산출
            timing_score, raw_score, _ = self._score_timing(grid_seg, raw_beats_seg)
            signed  = self._drift_signed(grid_seg, raw_beats_seg)
            t_label = self._timing_label(timing_score)
            d_label = self._drift_label(signed)

            chunk_result = {
                "index":       i,
                "measure":     measure_number,
                "half":        half_in_measure,
                "start_time":  round(t_start, 3),
                "end_time":    round(t_end,   3),
                "onset_count": int(len(grid_seg)),
                "beat_count":  int(len(raw_beats_seg)),  # 원본 madmom 검출 수
                "score":       round(timing_score, 3),
                "raw_score":   round(raw_score,    3),
                "tempo_label": t_label,
                "drift_label": d_label if d_label != "UNKNOWN" else "UNKNOWN",
            }
            print(json.dumps(chunk_result, ensure_ascii=False) + ",")
            results.append(chunk_result)

        return (json.dumps(results, ensure_ascii=False, indent=2),
                audio_start, beat_times, self.beat_grid, raw_beat_times)

    # ── Q-learning 기반 에이전트 루프 ──────────────────────────
    def run_agent(
        self,
        json_results: str,
        supervisor    = None,
        agent_id:     str = "rhythm_agent",
    ) -> list:
        """
        process() 결과를 받아 반 마디 chunk별로
        State 판별 → Q테이블 조회 → 액션 선택 →
        Reward 산출 → Q테이블 업데이트 → 슈퍼바이저 보고.

        CALL_SUPERVISOR 발동 조건
        ─────────────────────────
        fail_count 대신 Q값으로 판단.
        현재 State에서 best_action의 Q값이
        call_supervisor_q_threshold 이하이면 CALL_SUPERVISOR 선택.
        (학습 초기엔 Q값이 0.0이므로 룰베이스 기본 액션을 따르고,
         반복 실패로 Q값이 하락하면 자동으로 위임)

        파라미터
        ────────
        json_results : agent.process()가 반환한 JSON 문자열
        supervisor   : 슈퍼바이저 객체 (None이면 로컬 출력)
        agent_id     : 에이전트 식별자 (로그용)

        반환: chunk별 보고 결과 리스트
        """
        data             = json.loads(json_results)
        reports          = []
        prev_state       = "GOOD"
        prev_action      = None      # 첫 번째 reward는 null
        beat_interval_ms = self.beat_interval * 1000

        print("===== 박자 에이전트 루프 시작 (Q-learning, 반 마디 단위) =====")

        for idx, chunk in enumerate(data):
            curr_state = get_rhythm_state(chunk, beat_interval_ms)

            # chunk에 이미 index / measure / half 가 포함되어 있음 (process()에서 생성)
            chunk_index     = chunk["index"]
            measure_number  = chunk["measure"]
            half_in_measure = chunk["half"]

            # ── 1. Q테이블 조회 → best_action 및 Q값 확인 ────
            best_act  = self.q_table.best_action(curr_state)
            best_q    = self.q_table.get(curr_state, best_act)

            # Q값이 임계값 이하이면 CALL_SUPERVISOR (학습된 실패 패턴)
            if curr_state != "GOOD" and best_q <= self.call_supervisor_q_threshold:
                action = "CALL_SUPERVISOR"
                print(f"[Agent] Q[{curr_state}][{best_act}]={best_q:.4f} "
                      f"≤ {self.call_supervisor_q_threshold} → CALL_SUPERVISOR")
            else:
                action = best_act

            # ── 2. Reward 산출 ────────────────────────────────
            # 첫 번째 chunk는 직전 액션 없음 → None (JSON null)
            if prev_action is None:
                reward = None
            else:
                reward = compute_reward(prev_state, curr_state, prev_action)

            # ── 3. 다음 State 예측 ────────────────────────────
            if idx + 1 < len(data):
                next_state = get_rhythm_state(data[idx + 1], beat_interval_ms)
            else:
                next_state = curr_state

            # ── 4. Q테이블 업데이트 ───────────────────────────
            # Q(S,A) ← Q(S,A) + α[R + γ·maxQ(S',A') - Q(S,A)]
            # 첫 번째(reward=None)는 0.0으로 대체해 업데이트
            q_new = self.q_table.update(
                state      = curr_state,
                action     = action,
                reward     = reward if reward is not None else 0.0,
                next_state = next_state,
            )
            print(f"[Q-Update] chunk#{chunk_index:02d} "
                  f"마디{measure_number}-{'전' if half_in_measure==1 else '후'}반  "
                  f"({curr_state}, {action})  "
                  f"R={reward}  Q←{q_new:+.4f}")

            # ── 5. 슈퍼바이저 보고 ───────────────────────────
            report = report_to_supervisor(
                supervisor = supervisor,
                action     = action,
                curr_state = curr_state,
                reward     = reward,
                q_value    = q_new,
                measure    = measure_number,
                fail_count = None,          # fail_count 미사용
                meta       = {
                    "index":       chunk_index,
                    "half":        half_in_measure,   # 1=전반, 2=후반
                    "start_time":  chunk["start_time"],
                    "end_time":    chunk["end_time"],
                    "score":       chunk.get("score"),
                    "drift_label": chunk.get("drift_label"),
                },
            )
            reports.append(report)

            prev_state  = curr_state
            prev_action = action

        print("===== 박자 에이전트 루프 종료 =====\n")
        print(self.q_table.summary())
        return reports


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


def plot_beat_comparison(
    beat_grid, beat_times, audio_start=0.0,
    detection_stats=None, title="Beat Grid vs madmom",
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

    colors = {"grid":"#4A90D9","beat":"#E05C5C",
              "match":"#2ECC71","error_pos":"#E67E22","error_neg":"#8E44AD"}
    n_rows   = 4 if detection_stats else 3
    h_ratios = [2,2,3,2] if detection_stats else [2,2,3]
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4*n_rows),
                             gridspec_kw={"height_ratios": h_ratios})
    mode_str = "online" if REALTIME_MODE else "offline"
    fig.suptitle(f"{title} ({mode_str})", fontsize=14, fontweight="bold", y=0.99)
    x_min = min(shifted_grid[0], beat_times[0]) - 0.2
    x_max = max(shifted_grid[-1], beat_times[-1]) + 0.2

    ax1 = axes[0]
    ax1.set_title("beat 위치 타임라인", fontsize=11, pad=6)
    ax1.vlines(shifted_grid, 0.6, 1.4, color=colors["grid"],
               linewidth=1.5, alpha=0.8, label="Beat Grid")
    ax1.scatter(shifted_grid, np.ones(len(shifted_grid)),
                color=colors["grid"], s=40, zorder=3)
    ax1.vlines(beat_times, -0.4, 0.4, color=colors["beat"],
               linewidth=1.5, alpha=0.8, label=f"madmom ({mode_str})")
    ax1.scatter(beat_times, np.zeros(len(beat_times)),
                color=colors["beat"], s=40, zorder=3)
    for g, b in zip(matched_grid, matched_beat):
        ax1.plot([g,b],[1,0], color=colors["match"],
                 linewidth=0.8, alpha=0.5, linestyle="--")
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(["madmom","Beat Grid"], fontsize=10)
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
        ax3.axvline(0,             color="black",         linewidth=1.0)
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
        ax4.set_title(f"madmom 검출 정확도 ({mode_str})", fontsize=11, pad=6)
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
    print(f"[시각화] {out_path} 저장 완료")


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    MIDI_PATH  = "twinkle.mid"
    AUDIO_PATH = "reference.mp3"

    agent = ViolinRhythmAgent(MIDI_PATH)
    res, aligned_audio_start, beat_times_final, beat_grid_final, raw_beat_times_final = \
        agent.process(AUDIO_PATH)
    
    # 첫 10개 비교
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

    # ── Q-learning 에이전트 루프 ──────────────────────────────
    # Q테이블은 agent.q_table에 내장 (외부 주입 불필요)
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
