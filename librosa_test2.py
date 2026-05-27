import sys
print("실행 python:", sys.executable)

import numpy as np
import librosa
import pretty_midi
from scipy.spatial.distance import cdist
import subprocess
import os
import warnings
import json
import swift_f0
warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════
#  상수
# ══════════════════════════════════════════════════════════════
HOP_LENGTH    = 256          # 시간 해상도 (~5ms @ 44100Hz)
VIOLIN_FMIN   = librosa.note_to_hz('G3')   # 196 Hz  (개방 G선)
VIOLIN_FMAX   = librosa.note_to_hz('E7')   # 2637 Hz (최고음 근처)
CENTS_GOOD    = 50           # ±50cents 이내 = 좋은 음정 (반음의 절반)
ONSET_MIN_GAP = 0.08         # onset 최소 간격 80ms
NOISE_FLOOR   = 180.0        # Hz 이하는 바이올린 음역 밖 노이즈로 제거


# ══════════════════════════════════════════════════════════════
#  ViolinRhythmAgent
# ══════════════════════════════════════════════════════════════
class ViolinRhythmAgent:
    """
    바이올린 연주 평가 에이전트.

    평가 항목
    ─────────
    1. 타이밍 점수 : MIDI note onset  vs  검출 onset  (DTW 기반)
    2. 음정 점수   : MIDI note pitch  vs  pyin 피치   (cents 오차, 옥타브 fold)
    최종 점수      : 타이밍 × 0.5 + 음정 × 0.5
    """

    def __init__(self, midi_path: str, chunk_duration: float = 3.0):
        self.chunk_duration = chunk_duration
        self._prev_timing   = None
        self._prev_pitch    = None

        self.midi_notes = self._load_midi_notes(midi_path)
        print(f"[MIDI] {len(self.midi_notes)}개 음표 로드  "
              f"(첫 음: {self.midi_notes[0]['onset']:.3f}s "
              f"~ 마지막 음: {self.midi_notes[-1]['onset']:.3f}s)")

    # ──────────────────────────────────────────
    # MIDI 파싱 — 멜로디 라인(최고음)만 추출
    # ──────────────────────────────────────────
    def _load_midi_notes(self, midi_path: str):
        midi = pretty_midi.PrettyMIDI(midi_path)

        # 1) 전체 음표 수집
        all_notes = []
        for inst in midi.instruments:
            for n in inst.notes:
                all_notes.append({
                    "onset":  n.start,
                    "offset": n.end,
                    "pitch":  n.pitch,
                })
        all_notes.sort(key=lambda x: x["onset"])

        # 2) 동시 발음(onset 차이 30ms 이내) 중 최고음만 남김
        melody = []
        i = 0
        while i < len(all_notes):
            cluster = [all_notes[i]]
            j = i + 1
            while j < len(all_notes) and \
                  all_notes[j]["onset"] - all_notes[i]["onset"] < 0.03:
                cluster.append(all_notes[j])
                j += 1
            top = max(cluster, key=lambda x: x["pitch"])
            melody.append(top)
            i = j

        self._midi_notes_raw = melody
        return self._build_note_list(melody, octave_shift=0)

    def _build_note_list(self, raw_notes: list, octave_shift: int):
        """
        octave_shift: MIDI pitch에 더할 옥타브 수 (×12)
        note offset이 너무 짧으면(< 100ms) 다음 onset 직전까지 늘림
        """
        result = []
        for i, n in enumerate(raw_notes):
            p     = n["pitch"] + octave_shift * 12
            freq  = 440.0 * 2 ** ((p - 69) / 12)
            onset = n["onset"]
            offset = n["offset"]

            # offset이 onset+100ms 미만이면 늘림
            if offset < onset + 0.10:
                if i + 1 < len(raw_notes):
                    offset = min(raw_notes[i + 1]["onset"] - 0.02, onset + 0.40)
                else:
                    offset = onset + 0.40
                offset = max(offset, onset + 0.10)

            result.append({
                "onset":  onset,
                "offset": offset,
                "pitch":  p,
                "freq":   freq,
                "name":   pretty_midi.note_number_to_name(p),
            })
        return result

    def _auto_octave_shift(self, f0: np.ndarray,
                            voiced_flag: np.ndarray) -> int:
        """
        pyin 중앙값과 MIDI 중앙 주파수 비교 → 옥타브 차이 자동 계산
        """
        clean = f0[np.array(voiced_flag, dtype=bool)
                   & ~np.isnan(f0)
                   & (f0 > NOISE_FLOOR)]
        if len(clean) == 0:
            return 0

        pyin_med = float(np.median(clean))
        midi_med = float(np.median([n["freq"] for n in self.midi_notes]))

        if midi_med <= 0 or pyin_med <= 0:
            return 0

        shift = int(round(np.log2(pyin_med / midi_med)))
        print(f"[옥타브 보정] pyin중앙값={pyin_med:.1f}Hz  "
              f"MIDI중앙값={midi_med:.1f}Hz  → shift={shift:+d} 옥타브")
        return shift

    # ──────────────────────────────────────────
    # MP4 → WAV
    # ──────────────────────────────────────────
    def convert_mp4_to_wav(self, mp4_path: str) -> str:
        wav_path = mp4_path.replace(".mp4", ".wav")
        if not os.path.exists(wav_path):
            subprocess.run([
                "ffmpeg", "-i", mp4_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "1",
                wav_path
            ], check=True)
        return wav_path

    # ──────────────────────────────────────────
    # 연주 시작 시각 감지
    # ──────────────────────────────────────────
    def _detect_audio_start(self, y: np.ndarray, sr: int,
                             top_db: int = 28) -> float:
        _, idx = librosa.effects.trim(y, top_db=top_db)
        t = idx[0] / sr
        print(f"[시작점] 오디오 연주 시작 감지: {t:.3f}s")
        return t

    # ──────────────────────────────────────────
    # swiftf0 피치 추적
    # ──────────────────────────────────────────
    def _track_pitch(self, y: np.ndarray, sr: int):
        """
        swiftf0 기반 피치 추적.
        반환값을 pyin과 동일한 형태(f0 배열, voiced 마스크)로 맞춤.
        """

        # swiftf0는 float32, mono, 특정 sr 요구
        y32 = y.astype(np.float32)
        if sr != 16000:
            y32 = librosa.resample(y32, orig_sr=sr, target_sr=16000)
            sr_sf0 = 16000
        else:
            sr_sf0 = sr

        # swiftf0 실행 (hop_size는 ms 단위)
        hop_ms  = HOP_LENGTH / sr * 1000   # HOP_LENGTH 프레임 → ms 변환
        result  = swift_f0.compute(y32, sample_rate=sr_sf0, hop_size=hop_ms)

        freq        = np.array(result.frequency)   # Hz, 0이면 unvoiced
        voiced_flag = freq > NOISE_FLOOR           # 180Hz 이상만 voiced

        # swiftf0 시간축 → librosa HOP_LENGTH 기준 프레임으로 보간
        # (이후 _score_pitch에서 librosa.time_to_frames 써야 하므로)
        sf0_times    = np.arange(len(freq)) * (hop_ms / 1000)
        frame_times  = librosa.frames_to_time(
            np.arange(int(len(y) / HOP_LENGTH) + 1),
            sr=sr, hop_length=HOP_LENGTH
        )

        f0_interp     = np.interp(frame_times, sf0_times, freq)
        voiced_interp = (np.interp(frame_times, sf0_times,
                                    voiced_flag.astype(float)) > 0.5)

        print(f"[swiftf0] voiced 비율: "
            f"{voiced_interp.sum()/len(voiced_interp)*100:.1f}%")
        return f0_interp, voiced_interp

    # ──────────────────────────────────────────
    # 바이올린 onset 추출
    # ──────────────────────────────────────────
    def _extract_violin_onsets(self, y: np.ndarray, sr: int,
                                f0: np.ndarray,
                                voiced_flag: np.ndarray) -> np.ndarray:
        """
        A) pyin voiced 시작점  : 묵음→소리 전환
        B) pyin 피치 급변(80cents 이상) : 슬러 없는 음표 전환
        C) 에너지 onset (보조)
        """
        times = []
        prev_voiced = False
        prev_pitch  = None

        for i, (pitch, voiced) in enumerate(zip(f0, voiced_flag)):
            t = librosa.frames_to_time(i, sr=sr, hop_length=HOP_LENGTH)
            voiced = bool(voiced)

            if voiced and not prev_voiced:
                times.append(t)
            elif voiced and prev_voiced and prev_pitch is not None:
                if pitch > 0 and prev_pitch > 0:
                    if abs(1200 * np.log2(pitch / prev_pitch)) > 80:
                        times.append(t)

            prev_voiced = voiced
            prev_pitch  = pitch if voiced else None

        # 에너지 onset (보조)
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median
        )
        energy_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr, hop_length=HOP_LENGTH,
            backtrack=True, delta=0.25,
            wait=int(sr * ONSET_MIN_GAP / HOP_LENGTH),
            pre_max=3, post_max=3,
        )
        energy_times = librosa.frames_to_time(
            energy_frames, sr=sr, hop_length=HOP_LENGTH
        )

        all_times = np.array(sorted(set(times) | set(energy_times.tolist())))
        if len(all_times) > 1:
            keep = [all_times[0]]
            for t in all_times[1:]:
                if t - keep[-1] >= ONSET_MIN_GAP:
                    keep.append(t)
            all_times = np.array(keep)

        return all_times

    # ──────────────────────────────────────────
    # 타이밍 점수 (DTW)
    # ──────────────────────────────────────────
    def _score_timing(self, midi_onsets: np.ndarray,
                      audio_onsets: np.ndarray):
        if len(midi_onsets) < 2 or len(audio_onsets) < 2:
            return 0.0, []

        D, wp = librosa.sequence.dtw(
            C=cdist(midi_onsets.reshape(-1, 1),
                    audio_onsets.reshape(-1, 1))
        )
        wp     = np.array(wp[::-1])
        signed = [float(audio_onsets[j] - midi_onsets[i]) for i, j in wp]
        errors = [abs(e) for e in signed]

        raw   = float(np.exp(-np.mean(errors) * 5))
        score = raw if self._prev_timing is None \
                else 0.7 * self._prev_timing + 0.3 * raw
        self._prev_timing = score
        return score, signed

    # ──────────────────────────────────────────
    # 음정 점수 (pyin vs MIDI, 옥타브 fold)
    # ──────────────────────────────────────────
    def _score_pitch(self, midi_notes_seg: list,
                     f0: np.ndarray, voiced_flag: np.ndarray,
                     sr: int):
        cent_errors = []
        details     = []

        for note in midi_notes_seg:
            s_frame = librosa.time_to_frames(
                note["onset"],  sr=sr, hop_length=HOP_LENGTH)
            e_frame = librosa.time_to_frames(
                note["offset"], sr=sr, hop_length=HOP_LENGTH)
            s_frame = max(0, s_frame)
            e_frame = min(len(f0), e_frame)

            if s_frame >= e_frame:
                continue

            seg_f0 = f0[s_frame:e_frame]
            seg_v  = voiced_flag[s_frame:e_frame]

            # 노이즈 floor + NaN 제거
            valid = (np.array(seg_v, dtype=bool)
                     & ~np.isnan(seg_f0)
                     & (seg_f0 > NOISE_FLOOR))
            vp = seg_f0[valid]

            if len(vp) == 0:
                cent_errors.append(200.0)
                details.append({
                    "note": note["name"], "target_hz": round(note["freq"], 1),
                    "played_hz": None, "cents_error": 200.0, "ok": False,
                })
                continue

            mean_f0 = float(np.median(vp))
            if mean_f0 <= 0:
                continue

            # 옥타브 fold: -600~+600¢ 범위로 접기
            raw_cents    = 1200 * np.log2(mean_f0 / note["freq"])
            folded_cents = float(raw_cents - 1200 * round(raw_cents / 1200))
            cents_err    = float(abs(folded_cents))

            cent_errors.append(cents_err)
            details.append({
                "note":        note["name"],
                "target_hz":   round(note["freq"], 1),
                "played_hz":   round(mean_f0, 1),
                "cents_error": round(cents_err, 1),
                "ok":          bool(cents_err < CENTS_GOOD),
            })

        if not cent_errors:
            return 0.0, []

        note_scores = [1.0 if e < CENTS_GOOD
                       else float(np.exp(-(e - CENTS_GOOD) / 100))
                       for e in cent_errors]
        raw   = float(np.mean(note_scores))
        score = raw if self._prev_pitch is None \
                else 0.7 * self._prev_pitch + 0.3 * raw
        self._prev_pitch = score
        return score, details

    # ──────────────────────────────────────────
    # 라벨
    # ──────────────────────────────────────────
    @staticmethod
    def _timing_label(score: float) -> str:
        if score >= 0.80: return "정확  ✅"
        if score >= 0.55: return "보통  🔶"
        return                   "불안정 ❌"

    @staticmethod
    def _pitch_label(score: float) -> str:
        if score >= 0.80: return "음정 양호 ✅"
        if score >= 0.55: return "음정 불안정 🔶"
        return                   "음정 이탈 ❌"

    @staticmethod
    def _drift_label(signed_offsets: list) -> str:
        if not signed_offsets:
            return "측정불가"
        ms = float(np.mean(signed_offsets)) * 1000
        if   ms >  80: return f"늦게 연주 +{ms:.0f}ms"
        elif ms < -80: return f"빠르게 연주 {ms:.0f}ms"
        else:          return f"박자 정확 ({ms:+.0f}ms)"

    # ──────────────────────────────────────────
    # 메인
    # ──────────────────────────────────────────
    def process(self, audio_path: str) -> str:
        if audio_path.endswith(".mp4"):
            audio_path = self.convert_mp4_to_wav(audio_path)

        y, sr     = librosa.load(audio_path, sr=None, mono=True)
        total_dur = len(y) / sr
        print(f"[오디오] 길이: {total_dur:.2f}s  sr: {sr}")

        # 1. 연주 시작 시점 감지
        audio_start = self._detect_audio_start(y, sr)

        # 2. pyin 피치 추적
        print("[swiftf0] 피치 추적 중...")
        f0, voiced_flag = self._track_pitch(y, sr)
        print(f"[swiftf0] 완료. voiced 비율: "
              f"{voiced_flag.sum()/len(voiced_flag)*100:.1f}%")

        # 3. 옥타브 자동 보정
        shift = self._auto_octave_shift(f0, voiced_flag)
        if shift != 0:
            self.midi_notes = self._build_note_list(
                self._midi_notes_raw, octave_shift=shift)
            print(f"[옥타브 보정] {shift:+d} 옥타브 적용 완료")

        # 4. onset 추출
        all_onsets = self._extract_violin_onsets(y, sr, f0, voiced_flag)
        print(f"[onset] 검출: {len(all_onsets)}개 / MIDI 음표: {len(self.midi_notes)}개")

        # 5. MIDI onset을 audio_start만큼 shift
        shifted_notes = [
            {**n, "onset": n["onset"] + audio_start,
                  "offset": n["offset"] + audio_start}
            for n in self.midi_notes
        ]

        # 6. Chunk 단위 평가
        chunk_size   = int(sr * self.chunk_duration)
        total_chunks = len(y) // chunk_size
        results      = []

        print("\n===== 바이올린 연주 평가 시작 =====")

        for i in range(total_chunks):
            t_start = i * self.chunk_duration
            t_end   = (i + 1) * self.chunk_duration

            notes_seg        = [n for n in shifted_notes
                                 if t_start <= n["onset"] < t_end]
            midi_onsets_seg  = np.array([n["onset"] for n in notes_seg])
            audio_onsets_seg = all_onsets[
                (all_onsets >= t_start) & (all_onsets < t_end)]

            print(f"\n[{t_start:.1f}s ~ {t_end:.1f}s]  "
                  f"MIDI 음표: {len(notes_seg)}개  "
                  f"검출 onset: {len(audio_onsets_seg)}개")

            timing_score, signed_offsets = self._score_timing(
                midi_onsets_seg, audio_onsets_seg)
            pitch_score, pitch_details   = self._score_pitch(
                notes_seg, f0, voiced_flag, sr)
            final_score = timing_score * 0.5 + pitch_score * 0.5

            t_label = self._timing_label(timing_score)
            p_label = self._pitch_label(pitch_score)
            d_label = self._drift_label(signed_offsets)

            print(f"  타이밍: {timing_score:.3f}  {t_label}")
            print(f"  음정:   {pitch_score:.3f}  {p_label}")
            print(f"  밀림:   {d_label}")
            print(f"  최종:   {final_score:.3f}")

            bad = [d for d in pitch_details if not d["ok"]]
            if bad:
                print("  ⚠ 음정 이탈: "
                      + ", ".join(f"{d['note']}({d['cents_error']:.0f}¢)"
                                  for d in bad[:5]))

            results.append({
                "start_time":      round(t_start, 2),
                "end_time":        round(t_end, 2),
                "midi_note_count": len(notes_seg),
                "onset_count":     int(len(audio_onsets_seg)),
                "timing_score":    round(timing_score, 3),
                "pitch_score":     round(pitch_score, 3),
                "final_score":     round(final_score, 3),
                "timing_label":    t_label,
                "pitch_label":     p_label,
                "drift_label":     d_label,
                "pitch_details":   pitch_details,
            })

        return json.dumps(results, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
#  전체 성과 평가
# ══════════════════════════════════════════════════════════════
def evaluate_performance(json_results: str) -> dict:
    data = json.loads(json_results)
    if not data:
        return {"error": "결과 없음"}

    timing_avg = float(np.mean([c["timing_score"] for c in data]))
    pitch_avg  = float(np.mean([c["pitch_score"]  for c in data]))
    final_avg  = float(np.mean([c["final_score"]  for c in data]))

    if final_avg >= 0.80:
        level, recommend = "훌륭", "더 어려운 곡 도전 추천 🎯"
    elif final_avg >= 0.60:
        level, recommend = "적정", "현재 수준 유지 / 음정 집중 연습 권장"
    else:
        level, recommend = "미흡", "기초 스케일 / 느린 템포로 재연습 권장"

    weaknesses = []
    if timing_avg < 0.60: weaknesses.append("박자 안정성 부족")
    if pitch_avg  < 0.60: weaknesses.append("음정 정확도 부족")

    note_err: dict = {}
    for chunk in data:
        for d in chunk.get("pitch_details", []):
            if not d["ok"]:
                note_err[d["note"]] = note_err.get(d["note"], 0) + 1
    top_bad = sorted(note_err.items(), key=lambda x: -x[1])[:5]

    return {
        "overall_score":      round(final_avg, 3),
        "timing_avg":         round(timing_avg, 3),
        "pitch_avg":          round(pitch_avg, 3),
        "performance_level":  level,
        "recommendation":     recommend,
        "weaknesses":         weaknesses,
        "frequent_bad_notes": [{"note": n, "count": c} for n, c in top_bad],
    }


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agent = ViolinRhythmAgent("twinkle.mid", chunk_duration=3.0)
    res   = agent.process("performance.mp4")

    summary = evaluate_performance(res)
    print("\n=== 전체 성과 평가 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
