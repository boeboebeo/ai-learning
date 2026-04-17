import numpy as np
import matplotlib.pyplot as plt
import platform

# 한글 폰트 설정 (개선)
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False

# 폰트 제대로 로드되었는지 확인
import matplotlib.font_manager as fm
print("사용 가능한 한글 폰트:")
for font in fm.fontManager.ttflist:
    if 'Gothic' in font.name or 'Malgun' in font.name or 'Nanum' in font.name:
        print(f"  - {font.name}")


def demo_fft_complete():
    """
    원본 오디오 주파수 vs RMS 변화 주파수 비교
    """
    
    # ── 1. 파라미터 설정 ──
    duration = 5.0  # 5초
    sr = 44100
    hop = 512
    
    # 오디오 신호 생성: 440Hz (라 음) + 3Hz Tremolo
    t = np.linspace(0, duration, int(sr * duration))
    
    # 기본 음: 440Hz (A4)
    audio_freq = 440.0
    carrier = np.sin(2 * np.pi * audio_freq * t)
    
    # Tremolo: 3Hz로 진폭 변조
    tremolo_freq = 3.0
    tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * tremolo_freq * t)  # 0~1 범위
    
    # 최종 오디오: 440Hz 음에 3Hz Tremolo 적용
    y = carrier * tremolo
    
    
    # ── 2. 원본 오디오 FFT (음 주파수 분석) ──
    audio_fft = np.fft.rfft(y)
    audio_freqs = np.fft.rfftfreq(len(y), d=1/sr)
    audio_power = np.abs(audio_fft)
    
    
    # ── 3. RMS 추출 ──
    # 프레임 단위로 RMS 계산
    n_frames = int(len(y) / hop)
    rms = np.array([
        np.sqrt(np.mean(y[i*hop:(i+1)*hop]**2)) 
        for i in range(n_frames)
    ])
    rms_times = np.arange(len(rms)) * (hop / sr)
    
    
    # ── 4. RMS 변화 FFT (Tremolo 주파수 분석) ──
    rms_normalized = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
    rms_fft = np.fft.rfft(rms_normalized)
    rms_freqs = np.fft.rfftfreq(len(rms), d=(hop/sr))
    rms_power = np.abs(rms_fft)
    
    # 피크 찾기
    audio_peak_idx = np.argmax(audio_power[100:2000]) + 100  # 100Hz~2kHz 범위
    rms_peak_idx = np.argmax(rms_power[1:50]) + 1  # 1~10Hz 범위
    
    detected_audio_freq = audio_freqs[audio_peak_idx]
    detected_tremolo_freq = rms_freqs[rms_peak_idx]
    
    
    # ── 5. 시각화 ──
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)
    
    fig.suptitle('원본 오디오 주파수 vs Tremolo 주파수 비교', 
                 fontsize=16, fontweight='bold')
    
    
    # ═══ 왼쪽: 원본 오디오 분석 ═══
    
    # [1-1] 원본 오디오 파형 (일부만)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t[:4410], y[:4410], 'b-', linewidth=1)  # 처음 0.1초만
    ax1.set_title(f'[원본 오디오] 파형 (처음 0.1초)')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('amplitude')
    ax1.grid(True, alpha=0.3)
    # 텍스트 박스 위치 변경: 왼쪽 위 → 오른쪽 위
    ax1.text(0.98, 0.95, 
             f'{audio_freq}Hz 음파\n+ {tremolo_freq}Hz Tremolo',
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',  # 오른쪽 정렬
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=4)
    
    
    # [1-2] 원본 오디오 파형 (전체, Tremolo 보이게)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, y, 'b-', linewidth=0.5, alpha=0.7)
    # Tremolo envelope 그리기
    envelope = tremolo
    ax2.plot(t, envelope, 'r-', linewidth=2, label='Tremolo Envelope (3Hz)')
    ax2.plot(t, -envelope, 'r-', linewidth=2)
    ax2.set_title('[원본 오디오] 전체 파형 (Tremolo 효과 보임)')
    ax2.set_xlabel('시간 (초)')
    ax2.set_ylabel('진폭')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    
    # [1-3] 원본 오디오 FFT (전체 범위)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(audio_freqs[:5000], audio_power[:5000], 'purple', linewidth=1)
    ax3.axvline(detected_audio_freq, color='red', linestyle='--', linewidth=2, 
               label=f'감지: {detected_audio_freq:.1f}Hz')
    ax3.axvline(audio_freq, color='orange', linestyle=':', linewidth=2,
               label=f'실제: {audio_freq}Hz')
    ax3.set_title('[원본 오디오 FFT] 음 주파수 스펙트럼')
    ax3.set_xlabel('주파수 (Hz)')
    ax3.set_ylabel('강도')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim([0, 2000])
    
    
    # [1-4] 원본 오디오 FFT (확대)
    ax4 = fig.add_subplot(gs[3, 0])
    zoom_range = 100  # 440Hz 주변 ±100Hz
    zoom_start = max(0, int(audio_freq - zoom_range))
    zoom_end = int(audio_freq + zoom_range)
    zoom_indices = (audio_freqs >= zoom_start) & (audio_freqs <= zoom_end)
    
    ax4.plot(audio_freqs[zoom_indices], audio_power[zoom_indices], 'b-', linewidth=2)
    ax4.axvline(detected_audio_freq, color='red', linestyle='--', linewidth=2)
    ax4.set_title(f'[원본 오디오 FFT 확대] {audio_freq}Hz 주변')
    ax4.set_xlabel('주파수 (Hz)')
    ax4.set_ylabel('강도')
    ax4.grid(True, alpha=0.3)
    ax4.plot(detected_audio_freq, audio_power[audio_peak_idx], 'r*', 
            markersize=15, label=f'피크: {detected_audio_freq:.1f}Hz')
    ax4.legend(loc='upper right')
    
    
    # [1-5] 사이드밴드 (Tremolo로 인한 추가 주파수)
    ax5 = fig.add_subplot(gs[4, 0])
    sideband_range = 50
    sideband_indices = (audio_freqs >= audio_freq - sideband_range) & \
                       (audio_freqs <= audio_freq + sideband_range)
    ax5.plot(audio_freqs[sideband_indices], audio_power[sideband_indices], 
            'darkgreen', linewidth=2)
    ax5.axvline(audio_freq, color='red', linestyle='--', linewidth=1, 
               label=f'중심: {audio_freq}Hz')
    ax5.axvline(audio_freq - tremolo_freq, color='blue', linestyle=':', 
               linewidth=1, label=f'하측파대: {audio_freq - tremolo_freq:.1f}Hz')
    ax5.axvline(audio_freq + tremolo_freq, color='blue', linestyle=':', 
               linewidth=1, label=f'상측파대: {audio_freq + tremolo_freq:.1f}Hz')
    ax5.set_title('[사이드밴드] Tremolo가 만든 추가 주파수')
    ax5.set_xlabel('주파수 (Hz)')
    ax5.set_ylabel('강도')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8, loc='upper left')
    # 텍스트 박스 위치 변경
    ax5.text(0.98, 0.95,
             f'AM 변조 특성:\n중심 +/- {tremolo_freq}Hz',
             transform=ax5.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=9)
    
    
    # ═══ 오른쪽: RMS 변화 분석 (Tremolo) ═══
    
    # [2-1] RMS 시간 변화
    ax6 = fig.add_subplot(gs[0, 1])
    ax6.plot(rms_times, rms, 'r-', linewidth=2, label='RMS')
    # 이론적 Tremolo 그리기
    theoretical_tremolo = 0.5 + 0.5 * np.sin(2 * np.pi * tremolo_freq * rms_times)
    theoretical_tremolo = theoretical_tremolo * np.mean(rms)  # 스케일 맞춤
    ax6.plot(rms_times, theoretical_tremolo, 'g--', linewidth=2, 
            label=f'{tremolo_freq}Hz 이론값')
    ax6.set_title(f'[RMS 변화] Tremolo 효과')
    ax6.set_xlabel('시간 (초)')
    ax6.set_ylabel('RMS 값')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right')
    
    
    # [2-2] RMS 정규화
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.plot(rms_times, rms_normalized, 'g-', linewidth=2)
    ax7.set_title('[RMS 정규화] DC 제거')
    ax7.set_xlabel('시간 (초)')
    ax7.set_ylabel('정규화 값')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    
    # [2-3] RMS FFT (전체)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(rms_freqs[:100], rms_power[:100], 'purple', linewidth=2)
    ax8.axvline(detected_tremolo_freq, color='red', linestyle='--', linewidth=2,
               label=f'감지: {detected_tremolo_freq:.2f}Hz')
    ax8.axvline(tremolo_freq, color='orange', linestyle=':', linewidth=2,
               label=f'실제: {tremolo_freq}Hz')
    ax8.set_title('[RMS FFT] Tremolo 주파수 스펙트럼')
    ax8.set_xlabel('주파수 (Hz)')
    ax8.set_ylabel('강도')
    ax8.grid(True, alpha=0.3)
    ax8.legend(loc='upper right')
    ax8.set_xlim([0, 15])
    
    
    # [2-4] RMS FFT 확대
    ax9 = fig.add_subplot(gs[3, 1])
    ax9.stem(rms_freqs[:50], rms_power[:50], linefmt='b-', markerfmt='bo', basefmt=' ')
    ax9.axvline(detected_tremolo_freq, color='red', linestyle='--', linewidth=2)
    ax9.set_title('[RMS FFT 확대] 0-10Hz')
    ax9.set_xlabel('주파수 (Hz)')
    ax9.set_ylabel('강도')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim([0, 10])
    ax9.plot(detected_tremolo_freq, rms_power[rms_peak_idx], 'r*', 
            markersize=20, label=f'피크: {detected_tremolo_freq:.2f}Hz')
    # 텍스트 위치를 그래프 밖으로
    ax9.annotate(f'Tremolo 속도\n{detected_tremolo_freq:.2f}Hz',
                xy=(detected_tremolo_freq, rms_power[rms_peak_idx]),
                xytext=(detected_tremolo_freq + 2, rms_power[rms_peak_idx] * 0.7),
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax9.legend(loc='upper right')
    
    
    # [2-5] 주파수 비교 요약
    ax10 = fig.add_subplot(gs[4, 1])
    ax10.axis('off')
    
    summary_text = f"""분석 결과 요약
{'='*40}

원본 오디오 (음 높이):
  실제: {audio_freq} Hz
  감지: {detected_audio_freq:.1f} Hz
  오차: {abs(audio_freq - detected_audio_freq):.2f} Hz

Tremolo (진폭 변조):
  실제: {tremolo_freq} Hz
  감지: {detected_tremolo_freq:.2f} Hz
  오차: {abs(tremolo_freq - detected_tremolo_freq):.3f} Hz

{'─'*40}

핵심 차이:

- 원본 오디오 FFT:
  입력: y (원본 샘플)
  결과: {audio_freq}Hz (음 높이)
  범위: 0 ~ 22kHz

- RMS 변화 FFT:
  입력: rms (프레임별 크기)
  결과: {tremolo_freq}Hz (떨림 속도)
  범위: 0 ~ 10Hz

{'─'*40}

같은 FFT지만 입력이 다름!
"""
    
    ax10.text(0.05, 0.95, summary_text, 
             fontsize=9, family='monospace',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    
    plt.savefig('fft_complete_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
    # ── 콘솔 출력 ──
    print("=" * 70)
    print("FFT 완전 분석 데모")
    print("=" * 70)
    print()
    print("【원본 오디오】")
    print(f"  실제 주파수: {audio_freq} Hz")
    print(f"  FFT 감지: {detected_audio_freq:.1f} Hz")
    print(f"  오차: {abs(audio_freq - detected_audio_freq):.2f} Hz")
    print()
    print("【Tremolo (RMS 변화)】")
    print(f"  실제 주파수: {tremolo_freq} Hz")
    print(f"  FFT 감지: {detected_tremolo_freq:.2f} Hz")
    print(f"  오차: {abs(tremolo_freq - detected_tremolo_freq):.3f} Hz")
    print()
    print("=" * 70)
    print("핵심:")
    print(f"   • y의 FFT → {audio_freq}Hz (음 높이)")
    print(f"   • rms의 FFT → {tremolo_freq}Hz (떨림 속도)")
    print("   → 같은 FFT, 다른 입력, 다른 의미!")
    print("=" * 70)


if __name__ == "__main__":
    demo_fft_complete()