import sys
print("실행 python:", sys.executable)

import numpy as np
import librosa
import pretty_midi
from scipy.spatial.distance import cdist
import scipy.signal as signal
import subprocess
import os
import warnings
import json
warnings.filterwarnings("ignore", category=FutureWarning)

class RhythmAgentAdvanced:
    def __init__(self, midi_path, chunk_duration=1.0):
        self.midi_beats = self._load_midi_beats(midi_path)
        self.chunk_duration = chunk_duration
        self.prev_score = None
        self.prev_ratio = None
        self.prev_offset = None

    def convert_mp4_to_wav(self, mp4_path):
        wav_path = mp4_path.replace(".mp4", ".wav")
        if not os.path.exists(wav_path):
            command = [
                "ffmpeg", "-i", mp4_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "1",
                wav_path
            ]
            subprocess.run(command)
        return wav_path

    def _load_midi_beats(self, midi_path):
        midi = pretty_midi.PrettyMIDI(midi_path)
        tempo = midi.get_tempo_changes()[1][0]
        beat_interval = 60.0 / tempo
        end_time = midi.get_end_time()
        beats = np.arange(0, end_time, beat_interval)
        self.midi_bpm = tempo
        return beats

    def _expand_beats(self, total_duration):
        base = self.midi_beats
        length = base[-1]
        expanded = []
        t = 0
        while t < total_duration:
            expanded.extend(base + t)
            t += length
        return np.array(expanded)

    def _get_beat_segment(self, start, end):
        return self.midi_beats[
            (self.midi_beats >= start) & (self.midi_beats < end)
        ]

    def _extract_onsets(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        energy_onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            backtrack=False,
            delta=0.2,
            wait=8
        )
        energy_times = librosa.frames_to_time(energy_onsets, sr=sr)

        # energy onset 최소 간격 필터
        if len(energy_times) > 1:
            filtered_energy = [energy_times[0]]
            for t in energy_times[1:]:
                if t - filtered_energy[-1] > 0.15:
                    filtered_energy.append(t)
            energy_times = np.array(filtered_energy)

        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_track = []
        for i in range(pitches.shape[1]):
            idx = magnitudes[:, i].argmax()
            pitch_track.append(pitches[idx, i])
        pitch_track = np.array(pitch_track)
        pitch_smooth = signal.medfilt(pitch_track, kernel_size=5)
        pitch_delta = np.abs(np.diff(pitch_smooth, prepend=pitch_smooth[0]))
        pitch_thresh = np.mean(pitch_delta) + 0.5 * np.std(pitch_delta)
        pitch_onsets = np.where(pitch_delta > pitch_thresh)[0]
        pitch_times = librosa.frames_to_time(pitch_onsets, sr=sr)

        combined = np.unique(np.concatenate([energy_times, pitch_times]))
        if len(combined) < 2:
            combined = energy_times

        return combined, energy_times

    def _score_segment(self, beat_segment, audio_onsets):
        if len(audio_onsets) < 2 or len(beat_segment) < 2:
            return 0.0, [], []

        beat = beat_segment.reshape(-1, 1)
        audio = audio_onsets.reshape(-1, 1)
        dist = cdist(beat, audio, metric='euclidean')
        D, wp = librosa.sequence.dtw(C=dist)
        wp = np.array(wp[::-1])

        errors = [abs(beat_segment[i] - audio_onsets[j]) for i, j in wp]
        signed_offsets = [audio_onsets[j] - beat_segment[i] for i, j in wp]

        raw_score = np.exp(-np.mean(errors) * 5)
        smoothed = raw_score if self.prev_score is None else 0.7 * self.prev_score + 0.3 * raw_score
        self.prev_score = smoothed
        return smoothed, errors, signed_offsets

    # ✅ librosa.beat.tempo() 기반으로 교체
    def _estimate_bpm(self, y_chunk, sr):
        if len(y_chunk) < sr * 0.5:
            return None
        tempo = librosa.beat.tempo(y=y_chunk, sr=sr)
        return float(tempo[0])

    @staticmethod
    def _ratio_label(r):
        if r > 1.08:
            return f"FAST  (×{r:.2f})"
        elif r < 0.92:
            return f"SLOW  (×{r:.2f})"
        else:
            return f"STABLE(×{r:.2f})"

    @staticmethod
    def _offset_label(ms):
        if ms > 80:
            return f"BEHIND +{ms:.0f}ms (늦게 침)"
        elif ms < -80:
            return f"AHEAD  {ms:.0f}ms (빠르게 침)"
        else:
            return f"ON_TIME {ms:+.0f}ms"

    # ✅ y_chunk, sr 추가
    def _judge_performance(self, y_chunk, sr, signed_offsets):
        audio_bpm = self._estimate_bpm(y_chunk, sr)
        beat_bpm  = self.midi_bpm

        # ── 속도 판단 ──────────────────────────────
        if audio_bpm is None:
            if self.prev_ratio is not None:
                tempo_label = self._ratio_label(self.prev_ratio)
            else:
                tempo_label = "UNKNOWN"
        else:
            raw_ratio = audio_bpm / beat_bpm
            smooth_ratio = raw_ratio if self.prev_ratio is None \
                else 0.6 * self.prev_ratio + 0.4 * raw_ratio
            self.prev_ratio = smooth_ratio
            tempo_label = self._ratio_label(smooth_ratio)

        # ── 밀림 판단 ──────────────────────────────
        if len(signed_offsets) == 0:
            if self.prev_offset is not None:
                drift_label = self._offset_label(self.prev_offset)
            else:
                drift_label = "UNKNOWN"
        else:
            raw_offset_ms = np.mean(signed_offsets) * 1000
            smooth_offset = raw_offset_ms if self.prev_offset is None \
                else 0.6 * self.prev_offset + 0.4 * raw_offset_ms
            self.prev_offset = smooth_offset
            drift_label = self._offset_label(smooth_offset)

        return tempo_label, drift_label

    def process_stream(self, audio_path):
        if audio_path.endswith(".mp4"):
            audio_path = self.convert_mp4_to_wav(audio_path)

        y, sr = librosa.load(audio_path, sr=None)
        print(f"오디오 길이: {len(y)/sr:.2f}s, sr: {sr}")

        self.midi_beats = self._expand_beats(len(y) / sr)
        chunk_size = int(sr * self.chunk_duration)
        total_chunks = len(y) // chunk_size

        print("===== 실시간 박자 평가 시작 =====")

        results = []  # JSON용 리스트

        for i in range(total_chunks):
            chunk = y[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) < chunk_size:
                continue

            start_time = i * self.chunk_duration
            end_time   = (i + 1) * self.chunk_duration

            combined_onsets, _ = self._extract_onsets(chunk, sr)
            audio_onsets = combined_onsets + start_time
            beat_segment = self._get_beat_segment(start_time, end_time)

            print(f"[{start_time:.2f}s] onset: {len(audio_onsets)} / beat: {len(beat_segment)}")

            if len(audio_onsets) >= 2 and len(beat_segment) >= 2:
                score, errors, signed_offsets = self._score_segment(beat_segment, audio_onsets)
            else:
                score = self.prev_score * 0.95 if self.prev_score else 0.5
                signed_offsets = []

            tempo_label, drift_label = self._judge_performance(chunk, sr, signed_offsets)

            print(f"  안정화 박자 정확도: {score:.3f}")
            print(f"  ▶ 속도: {tempo_label}")
            print(f"  ▶ 밀림: {drift_label}")
            print("-" * 30)

            # ✅ JSON용 데이터 저장
            chunk_result = {
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "onset_count": int(len(audio_onsets)),
                "beat_count": int(len(beat_segment)),
                "score": round(score, 3),
                "tempo_label": tempo_label,
                "drift_label": drift_label
            }
            results.append(chunk_result)

        # 전체 결과 JSON 문자열
        json_output = json.dumps(results, ensure_ascii=False, indent=2)
        return json_output

def evaluate_performance(json_results):
        """
        전체 곡에 대한 성과 평가 및 난이도 추천
        json_results: process_stream 반환 JSON 문자열
        """
        data = json.loads(json_results)
        if not data:
            return {"overall_score": 0, "performance_level": "UNKNOWN", "recommend_harder": False}

        # 전체 평균 점수 계산
        scores = [chunk["score"] for chunk in data]
        avg_score = np.mean(scores)

        # 성과 판단 기준 (형님이 원하는 난이도 감각에 맞게 조정 가능)
        if avg_score < 0.5:
            level = "미흡"
            recommend = False
        elif avg_score < 0.75:
            level = "적정"
            recommend = True
        else:
            level = "훌륭"
            recommend = True

        result = {
            "overall_score": round(float(avg_score), 3),
            "performance_level": level,
            "recommend_harder": recommend
        }
        return result

if __name__ == "__main__":
    agent = RhythmAgentAdvanced("twinkle.mid")
    res = agent.process_stream("performance.mp4")
    print(res)
    performance_summary = evaluate_performance(res)
    print("\n=== 전체 성과 평가 ===")
    print(json.dumps(performance_summary, ensure_ascii=False, indent=2))

