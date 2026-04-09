import numpy as np
import librosa
from scipy.signal import savgol_filter, find_peaks
import os

def estimate_lpf_all_methods(y, sr):
    """
    모든 방법으로 LPF cutoff 추정 후 비교
    
    Returns:
    - results: dict with all methods' results
    - best_method: 추천 방법
    - best_cutoff: 추천 cutoff 값
    """
    
    # ===== 공통 준비 =====
    D = np.abs(librosa.stft(y, n_fft=4096, hop_length=2048))
        #n_fft = 4096. sr 전체 주파수를 4096개로 나누고
        #hop_length = 2048. 2048 / 22050(sr) = 0.093초 (93ms - 약 0.1초마다 한번씩 분석)
        # => Filter 분석하는거는 시간해상도가 중요하지 않으니까 hop_length크게함.
        # ADSR 분석시에는 더 작아야 하긴 함 . ex. 512
    spectrum = np.mean(D, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    
    min_freq = 200
    valid_mask = freqs >= min_freq
    spectrum_v = spectrum[valid_mask]
    freqs_v = freqs[valid_mask]
    
    # dB 변환
    spectrum_db = librosa.amplitude_to_db(spectrum_v, ref=np.max(spectrum_v))
        #가장 큰 값을 가진 오디오빈의 magnitude 를 max 로 잡고 -> 20*log10(max/max) = 20*log10(1) = 0dB 로 표현. 나머지는 모두 음수 
        #누군가의 0승은 항상 1
    # print(f"spectrum_v : {spectrum_v}")
    # print(f"spectrum_db : {spectrum_db}")
    
    results = {}
    
    # ===== Method 1: Slope-based (원래 A방식) =====
    try:
        slope = np.gradient(spectrum_v)
        slope_smooth = savgol_filter(slope, window_length=21, polyorder=3)
        skip_start = 20
        cutoff_idx_slope = skip_start + np.argmin(slope_smooth[skip_start:])
        cutoff_slope = freqs_v[cutoff_idx_slope]
        results['slope'] = {
            'cutoff': cutoff_slope,
            'index': cutoff_idx_slope,
            'status': 'success'
        }
    except Exception as e:
        results['slope'] = {'status': 'failed', 'error': str(e)}
    
    # ===== Method 2: -3dB Method (원래 B방식) =====
    try:
        max_idx = np.argmax(spectrum_v)
        max_db = spectrum_db[max_idx]
        target_db = max_db - 3
        
        after_peak = spectrum_db[max_idx:]
        valid_points = np.where(after_peak < target_db)[0]
        
        if len(valid_points) > 0:
            cutoff_idx_3db = max_idx + valid_points[0]
            cutoff_3db = freqs_v[cutoff_idx_3db]
            results['3db'] = {
                'cutoff': cutoff_3db,
                'index': cutoff_idx_3db,
                'status': 'success'
            }
        else:
            results['3db'] = {'status': 'failed', 'reason': 'no -3dB point found'}
    except Exception as e:
        results['3db'] = {'status': 'failed', 'error': str(e)}
    
    # ===== Method 3: Peak-based =====
    try:
        peaks, _ = find_peaks(spectrum_v, prominence=0.01)
        
        if len(peaks) > 0:
            last_peak = peaks[-1]
            slope = np.gradient(spectrum_v)
            slope_smooth = savgol_filter(slope, window_length=21, polyorder=3)
            
            search_range = min(50, len(slope_smooth) - last_peak - 1)
            
            if search_range > 0:
                cutoff_idx_peak = last_peak + np.argmin(slope_smooth[last_peak:last_peak+search_range])
                cutoff_peak = freqs_v[cutoff_idx_peak]
                results['peak'] = {
                    'cutoff': cutoff_peak,
                    'index': cutoff_idx_peak,
                    'peak_location': freqs_v[last_peak],
                    'status': 'success'
                }
            else:
                results['peak'] = {'status': 'failed', 'reason': 'no search range'}
        else:
            results['peak'] = {'status': 'failed', 'reason': 'no peaks found'}
    except Exception as e:
        results['peak'] = {'status': 'failed', 'error': str(e)}
    
    # ===== Method 4: Spectral Rolloff (95%) =====
    try:
        cumsum = np.cumsum(spectrum_v)
        total_energy = cumsum[-1]
        threshold = total_energy * 0.95
        
        rolloff_idx = np.where(cumsum >= threshold)[0]
        if len(rolloff_idx) > 0:
            cutoff_rolloff = freqs_v[rolloff_idx[0]]
            results['rolloff'] = {
                'cutoff': cutoff_rolloff,
                'index': rolloff_idx[0],
                'status': 'success'
            }
        else:
            results['rolloff'] = {'status': 'failed', 'reason': 'no rolloff point'}
    except Exception as e:
        results['rolloff'] = {'status': 'failed', 'error': str(e)}
    
    # ===== Method 5: High Frequency Content Threshold =====
    try:
        # 고주파 에너지가 급격히 떨어지는 지점
        high_freq_ratio = []
        window_size = 20
        
        for i in range(window_size, len(spectrum_v) - window_size):
            before = np.mean(spectrum_v[i-window_size:i])
            after = np.mean(spectrum_v[i:i+window_size])
            ratio = after / (before + 1e-8)
            high_freq_ratio.append(ratio)
        
        # ratio가 0.5 이하로 떨어지는 첫 지점
        threshold_idx = np.where(np.array(high_freq_ratio) < 0.5)[0]
        
        if len(threshold_idx) > 0:
            cutoff_idx_hfc = threshold_idx[0] + window_size
            cutoff_hfc = freqs_v[cutoff_idx_hfc]
            results['hfc'] = {
                'cutoff': cutoff_hfc,
                'index': cutoff_idx_hfc,
                'status': 'success'
            }
        else:
            results['hfc'] = {'status': 'failed', 'reason': 'no threshold crossing'}
    except Exception as e:
        results['hfc'] = {'status': 'failed', 'error': str(e)}
    
    # ===== Method 6: 2nd Derivative (가속도) =====
    try:
        slope = np.gradient(spectrum_v)
        slope_smooth = savgol_filter(slope, window_length=21, polyorder=3)
        slope2 = np.gradient(slope_smooth)
        slope2_smooth = savgol_filter(slope2, window_length=21, polyorder=3)
        
        skip_start = 20
        cutoff_idx_2nd = skip_start + np.argmin(slope2_smooth[skip_start:])
        cutoff_2nd = freqs_v[cutoff_idx_2nd]
        
        results['2nd_derivative'] = {
            'cutoff': cutoff_2nd,
            'index': cutoff_idx_2nd,
            'status': 'success'
        }
    except Exception as e:
        results['2nd_derivative'] = {'status': 'failed', 'error': str(e)}
    
    # ===== 성공한 방법들만 추출 =====
    successful = {k: v['cutoff'] for k, v in results.items() if v.get('status') == 'success'}
    
    # ===== 최적 방법 선택 =====
    if len(successful) > 0:
        # 중앙값 사용 (outlier에 덜 민감)
        best_cutoff = np.median(list(successful.values()))
        
        # 중앙값에 가장 가까운 방법 찾기
        best_method = min(successful.items(), key=lambda x: abs(x[1] - best_cutoff))[0]
    else:
        best_cutoff = None
        best_method = None
    
    return results, best_method, best_cutoff


# ===== 테스트 & 출력 =====
audio_files = [
    "Librosa-basics/audio_sample/saw+LPF(700).wav",
    "Librosa-basics/audio_sample/saw+LPF(5000hires).wav",
    "Librosa-basics/audio_sample/saw+LPF(300).wav",
    "Librosa-basics/audio_sample/saw+LPF(nofilter).wav",
    "Librosa-basics/audio_sample/noise+LPF(300).wav",
    "Librosa-basics/audio_sample/noise+LPF(1000).wav",
    "Librosa-basics/audio_sample/noise+LPF(5000hires).wav",
    "Librosa-basics/audio_sample/noise+LPF(5000res).wav",
    "Librosa-basics/audio_sample/square+LPF(nofilter).wav",
    "Librosa-basics/audio_sample/square+LPF(2000).wav",      
    "Librosa-basics/audio_sample/square+LPF(1100hires).wav", 
    "Librosa-basics/audio_sample/square+LPF(645mires).wav", 
]

print("=" * 80)
for path in audio_files:
    if not os.path.exists(path):
        continue
        
    filename = os.path.basename(path)
    
    # 실제 cutoff (파일명에서 추출)
    import re
    match = re.search(r'\((\d+)', filename)
    actual_cutoff = int(match.group(1)) if match else None
    
    y, sr = librosa.load(path)
    results, best_method, best_cutoff = estimate_lpf_all_methods(y, sr)
    
    print(f"\n[{filename}]")
    if actual_cutoff:
        print(f"  Actual Cutoff: {actual_cutoff}Hz")
    print(f"\n  Method Results:")
    
    for method, data in results.items():
        if data.get('status') == 'success':
            cutoff = data['cutoff']
            error = abs(cutoff - actual_cutoff) if actual_cutoff else None
            error_str = f" (error: {error:.0f}Hz)" if error else ""
            print(f"    {method:15s}: {cutoff:7.1f}Hz{error_str}")
        else:
            print(f"    {method:15s}: FAILED - {data.get('reason', data.get('error'))}")
    
    print(f"\n  ⭐ Best Method: {best_method}")
    print(f"  ⭐ Best Cutoff: {best_cutoff:.1f}Hz")
    
    if actual_cutoff and best_cutoff:
        final_error = abs(best_cutoff - actual_cutoff)
        accuracy = (1 - final_error/actual_cutoff) * 100
        print(f"  ⭐ Accuracy: {accuracy:.1f}%")
    
    print("-" * 80)

print("=" * 80)


# 모든 파일 처리 후 통계
def analyze_all_methods(audio_files):
    method_errors = {
        'slope': [],
        '3db': [],
        'peak': [],
        'rolloff': [],
        'hfc': [],
        '2nd_derivative': []
    }
    
    for path in audio_files:
        # 실제 cutoff
        match = re.search(r'\((\d+)', path)
        if not match:
            continue
        actual = int(match.group(1))
        
        y, sr = librosa.load(path)
        results, _, _ = estimate_lpf_all_methods(y, sr)
        
        for method, data in results.items():
            if data.get('status') == 'success':
                error = abs(data['cutoff'] - actual)
                method_errors[method].append(error)
    
    # 통계 출력
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    for method, errors in method_errors.items():
        if len(errors) > 0:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            success_rate = len(errors) / len(audio_files) * 100
            
            print(f"\n{method.upper()}:")
            print(f"  Mean Error  : {mean_error:.1f}Hz")
            print(f"  Std Dev     : {std_error:.1f}Hz")
            print(f"  Max Error   : {max_error:.1f}Hz")
            print(f"  Success Rate: {success_rate:.1f}%")

# 사용:
# analyze_all_methods(audio_files)