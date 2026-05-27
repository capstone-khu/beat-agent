import soundfile as sf
import librosa
import numpy as np

wav_path = "performance_22k.wav"

# BeatNet+가 내부적으로 어떻게 읽는지 확인
# 방법 1: soundfile
y_sf, sr_sf = sf.read(wav_path)
print(f"soundfile: shape={y_sf.shape}, sr={sr_sf}, duration={len(y_sf)/sr_sf:.2f}s")

# 방법 2: librosa (sr=None)
y_lb, sr_lb = librosa.load(wav_path, sr=None, mono=True)
print(f"librosa(sr=None): shape={y_lb.shape}, sr={sr_lb}, duration={len(y_lb)/sr_lb:.2f}s")

# BeatNet+ inference.py 직접 열어서 로드 방식 확인
import BeatNetPlus
import os
pkg_path = os.path.dirname(BeatNetPlus.__file__)
inf_path = os.path.join(pkg_path, "inference.py")
print(f"\nBeatNetPlus 경로: {pkg_path}")

with open(inf_path, "r") as f:
    for i, line in enumerate(f):
        if any(k in line for k in ["load", "read", "resample", "sample_rate", "sr"]):
            print(f"  {i+1:4d}: {line}", end="")