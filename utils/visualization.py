import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from config import TOLERANCE_MS, REALTIME_MODE


def _set_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                  "NanumBarunGothic", "Nanum Gothic", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rc("font", family=name)
            print(f"[폰트] {name} 적용")
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_korean_font()


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
