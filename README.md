## main.py 실행하시면 됩니다

**메타데이터 설명**
- note_count — chunk 내 MIDI 노트 수
- onset_count — grid beat 수
- beat_count — madmom 검출 beat 수
- raw_score — 비평활 타이밍 점수
EMA smoothing(직전 값과 현재 값을 가중 평균하는 방식으로, 과거 값이 현재 값에 계속 영향을 미치되 시간이 지날수록 영향이 줄어드는 평활화 기법)을 적용하기 전의 해당 chunk 순수 점수입니다. <br> score(EMA 적용)와 비교하면 이번 구간이 갑자기 나빠진 건지 추세적으로 나빠진 건지 구분할 수 있습니다.
raw_score가 score보다 훨씬 낮다  →  이번 구간만 갑자기 무너짐
raw_score ≈ score               →  추세적으로 안정/불안정
- tempo_label — ACCURATE / MODERATE / UNSTABLE<br>
score(EMA 점수)를 구간별로 레이블링한 값
- drift_ms — 평균 드리프트(ms), +늦음 / -빠름 / 0 정확
grid beat 대비 연주 beat의 평균 시간 오차
- beat_ratio — 템포 이탈 정도를 수치화한 값, beat_count / onset_count (FAST/SLOW 판별 기준값)
