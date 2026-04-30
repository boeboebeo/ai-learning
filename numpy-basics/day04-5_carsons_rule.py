import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel function

def fm_spectrum(delta_f, f_m, fc=0):
    """
    FM 신호의 주파수 스펙트럼 계산
    
    Parameters:
    -----------
    delta_f : float
        주파수 편이 (frequency deviation) in kHz
    f_m : float
        변조 신호의 최대 주파수 (modulating frequency) in kHz
    fc : float
        반송파 주파수 (carrier frequency) in kHz, 기본값 0 (상대 주파수)
    
    Returns:
    --------
    frequencies : array
        사이드밴드 주파수들
    amplitudes : array
        각 사이드밴드의 진폭 (Bessel 함수로 계산)
    """
    # 변조 지수 계산
    beta = delta_f / f_m
    
    # 사이드밴드 개수 결정 (베셀 함수가 충분히 작아질 때까지)
    max_n = int(beta + 6)  # 경험적으로 beta + 6개 정도면 충분
    
    frequencies = []
    amplitudes = []
    
    # 각 사이드밴드의 진폭 계산
    for n in range(-max_n, max_n + 1):
        freq = fc + n * f_m
        # Bessel 함수의 n차 값이 진폭
        amp = np.abs(jv(n, beta))
        
        frequencies.append(freq)
        amplitudes.append(amp)
    
    return np.array(frequencies), np.array(amplitudes), beta


def carsons_rule(delta_f, f_m):
    """
    Carson's Rule로 대역폭 계산
    
    BW ≈ 2(Δf + fm)
    """
    return 2 * (delta_f + f_m)


def plot_fm_spectrum(delta_f=75, f_m=15, fc=97300):
    """
    FM 스펙트럼 시각화
    
    Parameters:
    -----------
    delta_f : float
        주파수 편이 in kHz (기본값: 75 kHz, FM 라디오 표준)
    f_m : float
        변조 주파수 in kHz (기본값: 15 kHz, 오디오 최대 주파수)
    fc : float
        반송파 주파수 in kHz (기본값: 97.3 MHz = 97300 kHz)
    """
    
    # 스펙트럼 계산
    frequencies, amplitudes, beta = fm_spectrum(delta_f, f_m, fc)
    
    # Carson's Rule 대역폭
    bandwidth = carsons_rule(delta_f, f_m)
    
    # 그래프 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 상단: 전체 스펙트럼
    ax1.stem(frequencies, amplitudes, basefmt=' ')
    ax1.axvline(fc - bandwidth/2, color='red', linestyle='--', 
                label=f'Carson BW = {bandwidth:.0f} kHz', alpha=0.7)
    ax1.axvline(fc + bandwidth/2, color='red', linestyle='--', alpha=0.7)
    ax1.axvspan(fc - bandwidth/2, fc + bandwidth/2, alpha=0.1, color='red')
    
    ax1.set_xlabel('Frequency (kHz)')
    ax1.set_ylabel('Amplitude (normalized)')
    ax1.set_title(f'FM Spectrum: Δf={delta_f} kHz, fm={f_m} kHz, β={beta:.2f}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 하단: 반송파 근처 확대
    zoom_range = bandwidth * 1.5
    zoom_mask = (frequencies >= fc - zoom_range) & (frequencies <= fc + zoom_range)
    
    ax2.stem(frequencies[zoom_mask], amplitudes[zoom_mask], basefmt=' ')
    ax2.axvline(fc - bandwidth/2, color='red', linestyle='--', 
                label=f'Carson BW = {bandwidth:.0f} kHz', alpha=0.7)
    ax2.axvline(fc + bandwidth/2, color='red', linestyle='--', alpha=0.7)
    ax2.axvspan(fc - bandwidth/2, fc + bandwidth/2, alpha=0.1, color='red')
    
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel('Amplitude (normalized)')
    ax2.set_title('Zoomed view around carrier')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 정보 출력
    print(f"\n=== FM Signal Parameters ===")
    print(f"Carrier frequency (fc): {fc} kHz ({fc/1000:.1f} MHz)")
    print(f"Frequency deviation (Δf): {delta_f} kHz")
    print(f"Modulating frequency (fm): {f_m} kHz")
    print(f"Modulation index (β): {beta:.2f}")
    print(f"\n=== Carson's Rule ===")
    print(f"Bandwidth (BW): {bandwidth:.0f} kHz")
    print(f"BW = 2(Δf + fm) = 2({delta_f} + {f_m}) = {bandwidth:.0f} kHz")
    print(f"\nChannel spacing needed: ~{np.ceil(bandwidth/10)*10:.0f} kHz")
    
    plt.show()
    
    return frequencies, amplitudes, beta, bandwidth


def compare_modulation_indices():
    """
    여러 변조 지수에 대한 스펙트럼 비교
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # 다양한 β 값 테스트
    test_cases = [
        (15, 15, "β=1 (Narrowband FM)"),
        (30, 15, "β=2"),
        (75, 15, "β=5 (FM Radio)"),
        (150, 15, "β=10 (Wideband FM)")
    ]
    
    for idx, (delta_f, f_m, title) in enumerate(test_cases):
        frequencies, amplitudes, beta = fm_spectrum(delta_f, f_m, fc=0)
        bandwidth = carsons_rule(delta_f, f_m)
        
        axes[idx].stem(frequencies, amplitudes, basefmt=' ')
        axes[idx].axvline(-bandwidth/2, color='red', linestyle='--', alpha=0.7)
        axes[idx].axvline(bandwidth/2, color='red', linestyle='--', alpha=0.7)
        axes[idx].axvspan(-bandwidth/2, bandwidth/2, alpha=0.1, color='red')
        
        axes[idx].set_xlabel('Frequency offset from fc (kHz)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].set_title(f'{title}\nΔf={delta_f} kHz, BW={bandwidth:.0f} kHz')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(-200, 200)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 예제 1: FM 라디오 표준 (97.3 MHz)
    print("Example 1: FM Radio Standard (97.3 MHz)")
    plot_fm_spectrum(delta_f=75, f_m=15, fc=97300)
    
    # 예제 2: 여러 변조 지수 비교
    print("\nExample 2: Comparing different modulation indices")
    compare_modulation_indices()
    
    # 예제 3: 스테레오 FM (더 높은 변조 주파수)
    print("\nExample 3: Stereo FM with pilot tone")
    plot_fm_spectrum(delta_f=75, f_m=53, fc=97300)  # 53 kHz = 스테레오 부반송파 포함