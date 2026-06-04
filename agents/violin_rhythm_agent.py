import json
import os
import subprocess

import librosa
import numpy as np
import pretty_midi

from config import ALPHA, GAMMA, STATES, ACTIONS
from core.beat_utils import (
    run_madmom,
    build_beat_grid_from_midi,
    build_chunks_from_score_metadata,
    align_to_grid,
    resample_to_midi_bpm,
)
from core.reward import compute_reward
from core.state import get_rhythm_state
from agents.q_table import RhythmQTable
from agents.supervisor import report_to_supervisor


class ViolinRhythmAgent:
    def __init__(self, midi_path: str,
                 beats_per_measure: int = 4,
                 score_metadata_path: str | None = None):
        self.midi_path               = midi_path
        self.beats_per_measure       = beats_per_measure
        self.beats_per_half_measure  = beats_per_measure // 2
        self._prev_timing            = None
        self.score_metadata_path     = score_metadata_path

        midi               = pretty_midi.PrettyMIDI(midi_path)
        self.bpm           = midi.get_tempo_changes()[1][0]
        self.beat_interval = 60.0 / self.bpm
        self.half_measure_duration = self.beat_interval * self.beats_per_half_measure
        self.midi_notes    = self._load_midi_notes(midi_path)
        self.beat_grid     = build_beat_grid_from_midi(midi_path)

        if score_metadata_path:
            self.score_chunks = build_chunks_from_score_metadata(score_metadata_path)
        else:
            self.score_chunks = None

        self.q_table = RhythmQTable(STATES, ACTIONS, alpha=ALPHA, gamma=GAMMA)

        print(f"[Beat Grid] 첫 박: {self.beat_grid[0]:.3f}s  "
              f"마지막 박: {self.beat_grid[-1]:.3f}s  "
              f"BPM: {self.bpm:.1f}  간격: {self.beat_interval*1000:.0f}ms")
        print(f"[반 마디]   {self.beats_per_half_measure}박 "
              f"= {self.half_measure_duration*1000:.0f}ms")
        print(f"[Chunk 기준] {'score_metadata JSON' if score_metadata_path else 'beat_grid (MIDI)'}")
        print(f"[Q-Table]  초기화 완료 — States: {STATES}, Actions: {ACTIONS}")
        print(f"[Supervisor 위임] CALL_SUPERVISOR Q값이 타 액션보다 높을 때 자동 선택")

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
        if len(grid_onsets) == 0 or len(beat_times) == 0:
            prev = self._prev_timing if self._prev_timing is not None else 0.0
            return prev, 0.0, []

        signed = []
        for g in grid_onsets:
            diffs = np.abs(beat_times - g)
            ri    = int(np.argmin(diffs))
            signed.append(float(beat_times[ri] - g))

        mean_drift = float(np.mean(signed))
        residuals  = [s - mean_drift for s in signed]
        rel        = [abs(r) / self.beat_interval for r in residuals]
        raw        = float(np.exp(-np.mean(rel) * 3))

        smoothed = raw if self._prev_timing is None \
                   else 0.5 * self._prev_timing + 0.5 * raw
        self._prev_timing = smoothed
        return smoothed, raw, signed

    def _drift_signed(self, grid_onsets, raw_beat_times):
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

        if self.score_chunks is not None:
            chunk_defs = self.score_chunks
        else:
            half = self.beats_per_half_measure
            half_measure_starts = self.beat_grid[::half] + audio_start
            chunk_defs = []
            for i, ts in enumerate(half_measure_starts):
                te = (half_measure_starts[i + 1]
                      if i + 1 < len(half_measure_starts)
                      else beat_times[-1] + self.beat_interval)
                chunk_defs.append({
                    "measure": i // 2 + 1,
                    "half":    i % 2 + 1,
                    "start":   float(ts),
                    "end":     float(te),
                })

        results = []
        print("\n===== 바이올린 박자 평가 시작 (반 마디 단위) =====\n")

        for i, cdef in enumerate(chunk_defs):
            t_start         = cdef["start"]
            t_end           = cdef["end"]
            measure_number  = cdef["measure"]
            half_in_measure = cdef["half"]

            grid_seg      = shifted_grid[
                (shifted_grid >= t_start) & (shifted_grid < t_end)]
            beats_seg     = beat_times[
                (beat_times  >= t_start) & (beat_times  < t_end)]
            raw_beats_seg = raw_beat_times[
                (raw_beat_times >= t_start) & (raw_beat_times < t_end)]

            timing_score, raw_score, _ = self._score_timing(grid_seg, raw_beats_seg)
            signed  = self._drift_signed(grid_seg, raw_beats_seg)
            t_label = self._timing_label(timing_score)
            d_label = self._drift_label(signed)

            note_count = cdef.get("note_count", len(grid_seg))

            chunk_result = {
                "index":       i,
                "measure":     measure_number,
                "half":        half_in_measure,
                "start_time":  round(t_start, 3),
                "end_time":    round(t_end,   3),
                "note_count":  note_count,
                "onset_count": int(len(grid_seg)),
                "beat_count":  int(len(raw_beats_seg)),
                "score":       round(timing_score, 3),
                "raw_score":   round(raw_score,    3),
                "tempo_label": t_label,
                "drift_label": d_label if d_label != "UNKNOWN" else "UNKNOWN",
            }
            print(json.dumps(chunk_result, ensure_ascii=False) + ",")
            results.append(chunk_result)

        return (json.dumps(results, ensure_ascii=False, indent=2),
                audio_start, beat_times, self.beat_grid, raw_beat_times)

    def run_agent(
        self,
        json_results: str,
        supervisor    = None,
        agent_id:     str = "rhythm_agent",
    ) -> list:
        data             = json.loads(json_results)
        reports          = []
        prev_state       = "GOOD"
        prev_action      = None
        beat_interval_ms = self.beat_interval * 1000

        print("===== 박자 에이전트 루프 시작 (Q-learning, 반 마디 단위) =====")

        for idx, chunk in enumerate(data):
            curr_state = get_rhythm_state(chunk, beat_interval_ms)

            chunk_index     = chunk["index"]
            measure_number  = chunk["measure"]
            half_in_measure = chunk["half"]

            onset_count = chunk.get("onset_count", 0)
            beat_count  = chunk.get("beat_count",  0)
            beat_ratio  = round(beat_count / onset_count, 3) if onset_count > 0 else None

            drift_label = chunk.get("drift_label", "UNKNOWN")
            drift_ms: float | None = None
            if "LATE" in drift_label:
                try:
                    drift_ms = float(drift_label.split("+")[-1].replace("ms)", "").replace("ms", ""))
                except ValueError:
                    drift_ms = None
            elif "EARLY" in drift_label:
                try:
                    drift_ms = -abs(float(drift_label.split("-")[-1].replace("ms)", "").replace("ms", "")))
                except ValueError:
                    drift_ms = None
            else:
                drift_ms = 0.0

            action   = self.q_table.best_action(curr_state)
            action_q = self.q_table.get(curr_state, action)
            print(f"[Agent] chunk#{chunk['index']:02d}  state={curr_state}  "
                  f"→ action={action}  Q={action_q:+.4f}")

            reward = None if prev_action is None else \
                     compute_reward(prev_state, curr_state, prev_action)

            next_state = get_rhythm_state(data[idx + 1], beat_interval_ms) \
                         if idx + 1 < len(data) else curr_state

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

            report = report_to_supervisor(
                supervisor = supervisor,
                action     = action,
                curr_state = curr_state,
                reward     = reward,
                q_value    = q_new,
                measure    = measure_number,
                meta       = {
                    "index":       chunk_index,
                    "half":        half_in_measure,
                    "start_time":  chunk["start_time"],
                    "end_time":    chunk["end_time"],
                    "note_count":  chunk.get("note_count"),
                    "onset_count": onset_count,
                    "beat_count":  beat_count,
                    "score":       chunk.get("score"),
                    "raw_score":   chunk.get("raw_score"),
                    "tempo_label": chunk.get("tempo_label"),
                    "drift_label": drift_label,
                    "drift_ms":    drift_ms,
                    "beat_ratio":  beat_ratio,
                },
            )
            reports.append(report)

            prev_state  = curr_state
            prev_action = action

        print("===== 박자 에이전트 루프 종료 =====\n")
        print(self.q_table.summary())
        return reports
