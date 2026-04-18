# day 08 - 1 modulation_estimation 

import librosa
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import platform

# 한글 폰트 설정
system = platform.system()
if system == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def detect_modulation(y, sr, hop=256):
    #Modulation detection
    
    #Feature(: 특징, 측정값. RMS 는 1가지 특성을 가진것) 추출
        #MFCC = 13가지 특성(음색 계수)를 가진것 : feature 13개 
    rms = librosa.feature.rms(y=y, hop_length=hop)[0] 
        # [0] : librosa의 모든 feature는 (n_features, n_frames) 으로 반환되기 때문에 -> (n_features, ) 로 바꾸는 것 
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        #소리의 무게중심 : Centroid = Σ(주파수 × 강도) / Σ(강도)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
        # 0점을 지나는 횟수(+/- 바뀌는 횟수) / 전체샘플수 

    # print(rms)

    #------Tremolo detection------
    print(f"\n [Tremolo Analyze]")
    
    #RMS 의 주기적인 변화 
    rms_normalized = (rms - np.mean(rms)) / (np.std(rms)+1e-8) 
        #1. 각 rms 에서 평균 rms 를 빼고 (음수 나올수도 있음) => 이제 평균이 0 이 되는것 (DC 성분을 제거함. 원래는 다 RMS 수치라서 + 였음)
        #2. 각각을 표준편차로 나누기(Scale) . 원본의 진폭이 크던 작던 표준화 하는것 => 변화패턴에만 집중하기 위해서
        # => 얼마나 크게 흔들리는지 말고, 얼마나 자주 흔들리는지만 보기위해서 rms_normalized 를 함 (DC 제거. )

    #FFT 로 변조 주파수 찾기
    rms_fft = np.fft.rfft(rms_normalized)
        #rfft = Real FFT (실수 : Real 입력 전용). 출력은 N개에서 N개 복소수
        #fft = input에 복소수 가능함. 출력은 N개에서 N/2 + 1 개 복소수가 출력됨 (실수신호는 주파수가 대칭이라서. Nyquist 주파수 까지만)
    rms_freqs = np.fft.rfftfreq(len(rms), d=(hop/sr)) 
        #각 fft 결과가 몇 Hz인지
        #(입력길이, d=샘플 간격) . d = hop/sr = 512/44100 = 0.0116초 => 주파수 배열 생성
    rms_power = np.abs(rms_fft)
        #rms_fft 그냥 절대값 취한것

    # print(rms_fft[:10])
    # print(rms_freqs[:10])
    # print(times[:10])
    # print(len(rms_fft))
    # print(sr)
    # print(len(rms))
    # print(rms_power[:10])

    # print(rms_freqs[1:20]) 
        # 대략 0.498Hz ~ 9.474Hz 정도 출력되게 됨
        # 아래의 rms_power[1:20]에서는 그 0.498Hz ~ 9.47Hz 사이의 주파수에서의 최고 레벨을 찾음
            # => 그러면 9.47Hz 이내의 속도를 가진 tremolo 의 피크값을 구할 수 있음


    #DC 제외하고 가장 강한 주파수
    if len(rms_power) > 1: 
        # 이론적으로는 항상 95개라 불필요하지만 (방어코드)오디오가 너무 짧으면 프레임이 1개 이하일 수 있음
        peak_idx = np.argmax(rms_power[1:20]) + 1 
            # 인덱스 1 부터 시작하니까 +1 하기
        tremolo_rate = rms_freqs[peak_idx]
        tremolo_strength = rms_power[peak_idx] / np.sum(rms_power[1:20])

        #Tremolo 판정 (변조 속도 1~10Hz, 강도 0.3 이상)
        if 1 < tremolo_rate < 10 and tremolo_strength > 0.3:
            tremolo_detected = True
            print(f"    ✅ Tremolo 감지 : {tremolo_rate:.2f}Hz (강도 : {tremolo_strength:.2f})")
        else:
            tremolo_detected = False
            print(f"    ❌ Tremolo 미감지")
    
    else:
        tremolo_detected = False
        tremolo_rate = 0.0
        tremolo_strength = 0.0
        print(f"    ❌Tremolo 미감지 (신호 짧음)")


#(1)------Vibrato detection------
    print(f"\n  [Vibrato Analyze]")

    #spectral Centroid 의 주기적 변화
    centroid_normalized = (centroid - np.mean(centroid)) / (np.std(centroid)+ 1e-8)

    # FFT 로 변조 주파수 찾기
    centroid_fft = np.fft.rfft(centroid_normalized)
    centroid_freqs = np.fft.rfftfreq(len(centroid), d=(hop/sr))
    centroid_power = np.abs(centroid_fft)
        #근데 현재 6Hz vibrato 인데, 3.49로 나오는애가 하나있음. sawtooth 고조파 때문일 수도 있으니 코드를 바꿔보자
        # => 상위 3개의 피크를 찾고 
        # => 가장 높은 주파수가 다른 것의 정수배인지 확인

    top_3_indices = np.argsort(centroid_power[1:20])[-3:] + 1
    top_3_freqs = centroid_freqs[top_3_indices]
    top_3_powers = centroid_power[top_3_indices]

    highest_freq = np.max(top_3_freqs)

    #DC 제외하고 가장 강한 주파수
    if len(centroid_power) > 1:
        # peak_idx = np.argmax(centroid_power[1:15]) + 1 # 1 ~ 15Hz 범위에서 
        vibrato_rate = centroid_freqs[peak_idx]
        vibrato_strength = centroid_power[peak_idx] / np.sum(centroid_power[1:15])

        #Vibrato 판정 (변조속도 4~8kHz, 강도 0.2 이상)
        if 1 < vibrato_rate < 8 and vibrato_strength > 0.1:
            vibrato_detected = True
            print(f"    ✅ Vibrato 감지: {vibrato_rate:.2f} Hz (강도: {vibrato_strength:.2f})")
        
        else :
            vibrato_detected = False
            print(f"    ❌ Vibrato 미감지")

    else:
        vibrato_detected = False
        vibrato_rate = 0.0
        vibrato_strength = 0.0
        print(f"    ❌ Vibrato 미감지 (신호 짧음)")

#(2)------Vibrato detection------
    # print(f"\n  [Vibrato Analyze]")

    # # Spectral Centroid 의 주기적 변화
    # centroid_normalized = (centroid - np.mean(centroid)) / (np.std(centroid) + 1e-8)  # 괄호 수정!

    # # FFT 로 변조 주파수 찾기
    # centroid_fft = np.fft.rfft(centroid_normalized)
    # centroid_freqs = np.fft.rfftfreq(len(centroid), d=(hop/sr))
    # centroid_power = np.abs(centroid_fft)

    # # DC 제외하고 가장 강한 주파수
    # if len(centroid_power) > 1:
    #     # 4Hz 이상에서만 탐색 (서브하모닉 제거)
    #     min_vibrato_hz = 4.0
    #     min_idx = max(1, np.searchsorted(centroid_freqs, min_vibrato_hz))
        
    #     # 상위 3개 피크 찾기
    #     search_range = min(50, len(centroid_power))
    #     top_3_indices = np.argsort(centroid_power[min_idx:search_range])[-3:] + min_idx
    #     top_3_freqs = centroid_freqs[top_3_indices]
    #     top_3_powers = centroid_power[top_3_indices]
        
    #     # 가장 강한 피크 선택
    #     strongest_idx = top_3_indices[np.argmax(top_3_powers)]
    #     vibrato_rate = centroid_freqs[strongest_idx]
        
    #     # 강도 계산 (4~12Hz 범위에서)
    #     vibrato_range_end = min(search_range, np.searchsorted(centroid_freqs, 12.0))
    #     vibrato_strength = centroid_power[strongest_idx] / np.sum(centroid_power[min_idx:vibrato_range_end])
        
    #     # 디버깅 출력
    #     print(f"    [디버깅] 상위 피크:")
    #     for idx in reversed(top_3_indices):
    #         print(f"      {centroid_freqs[idx]:.2f}Hz: {centroid_power[idx]:.1f}")
        
    #     # Vibrato 판정 (변조속도 4~10Hz, 강도 0.2 이상)
    #     if 4 < vibrato_rate < 10 and vibrato_strength > 0.2:
    #         vibrato_detected = True
    #         print(f"    ✅ Vibrato 감지: {vibrato_rate:.2f} Hz (강도: {vibrato_strength:.2f})")
    #     else:
    #         vibrato_detected = False
    #         print(f"    ❌ Vibrato 미감지 (rate={vibrato_rate:.2f}Hz, strength={vibrato_strength:.2f})")

    # else:
    #     vibrato_detected = False
    #     vibrato_rate = 0.0
    #     vibrato_strength = 0.0
    #     print(f"    ❌ Vibrato 미감지 (신호 짧음)")



#(3)Vibrato detection 부분 수정
    #------Vibrato detection------
    # print(f"\n  [Vibrato Analyze]")

    # # F0 (Pitch) 추출 사용 (Centroid 대신)
    # try:
        f0 = librosa.yin(y, fmin=80, fmax=800, sr=sr, hop_length=hop)
        
    #     # F0가 0인 부분 제거 (무음 구간)
    #     f0_valid = f0[f0 > 0]
        
    #     if len(f0_valid) > 20:  # 충분한 데이터가 있을 때
    #         # F0 정규화
    #         f0_normalized = (f0_valid - np.mean(f0_valid)) / (np.std(f0_valid) + 1e-8)
            
    #         # FFT
    #         f0_fft = np.fft.rfft(f0_normalized)
    #         f0_freqs = np.fft.rfftfreq(len(f0_valid), d=(hop/sr))
    #         f0_power = np.abs(f0_fft)
            
    #         # 4Hz 이상에서 피크 찾기 (Vibrato는 보통 4Hz 이상)
    #         min_idx = max(1, np.searchsorted(f0_freqs, 4.0))
    #         max_idx = min(len(f0_power), np.searchsorted(f0_freqs, 15.0))
            
    #         if max_idx > min_idx:
    #             peak_idx = np.argmax(f0_power[min_idx:max_idx]) + min_idx
    #             vibrato_rate = f0_freqs[peak_idx]
    #             vibrato_strength = f0_power[peak_idx] / np.sum(f0_power[min_idx:max_idx])
                
    #             # Vibrato 판정 (변조속도 4~12Hz, 강도 0.15 이상)
    #             if 4 < vibrato_rate < 12 and vibrato_strength > 0.15:
    #                 vibrato_detected = True
    #                 print(f"    ✅ Vibrato 감지: {vibrato_rate:.2f} Hz (강도: {vibrato_strength:.2f})")
    #             else:
    #                 vibrato_detected = False
    #                 print(f"    ❌ Vibrato 미감지 (rate={vibrato_rate:.2f}Hz, strength={vibrato_strength:.2f})")
    #         else:
    #             vibrato_detected = False
    #             vibrato_rate = 0.0
    #             vibrato_strength = 0.0
    #             f0_fft = np.array([0])
    #             f0_freqs = np.array([0])
    #             print(f"    ❌ Vibrato 미감지 (주파수 범위 부족)")
    #     else:
    #         vibrato_detected = False
    #         vibrato_rate = 0.0
    #         vibrato_strength = 0.0
    #         f0_fft = np.array([0])
    #         f0_freqs = np.array([0])
    #         print(f"    ❌ Vibrato 미감지 (유효 F0 부족)")

    # except Exception as e:
    #     print(f"    ❌ Vibrato 분석 실패: {e}")
    #     vibrato_detected = False
    #     vibrato_rate = 0.0
    #     vibrato_strength = 0.0
    #     f0_fft = np.array([0])
    #     f0_freqs = np.array([0])



# ── 일반 AM/FM 감지 ──
    # print(f"\n  [Modulation Analyze]")
    
    # # AM (진폭 변조)
    # rms_variation = np.std(np.diff(rms))
    #     #np.diff() : 차분방정식 / np.std( ) : 표준편차
    #     #np.diff() : rms[t+1] - rms[t] : 볼륨이 얼마나 빠르게 변화하는지
    #     #np.std() : 변화량의 흔들림 정도 
    #         # AM 없으면 diff 거의 0. volume : ---------- 이렇게 신호가 유지됨 

    # if rms_variation > 0.01: #std(diff) 가 작다면 거의 레벨이 일정하다는 뜻
    #     am_detected = True
    #     print(f"    ✅ AM 감지: 변화량 {rms_variation:.4f}")
    # else:
    #     am_detected = False
    #     print(f"    ❌ AM 미감지")
    
    # # FM (주파수 변조)
    # centroid_variation = np.std(np.diff(centroid))
    #     #centroid(소리의 무게중심을 구해서) -> diff 후 -> std 처리
    #     # FM 없으면 centroid 거의 0.centorid : ---------- 
    # if centroid_variation > 100: # 주파수의 무게중심이 자주 바뀐다면은 fm 존재하는것 
    #     fm_detected = True
    #     print(f"    ✅ FM 감지: 변화량 {centroid_variation:.1f} Hz")
    # else:
    #     fm_detected = False
    #     print(f"    ❌ FM 미감지")  



#(수정)sideband 기반 AM/FM 감지 추가
    print(f"\n [Sideband Analysis]")

    #STFT 
    D = librosa.stft(y, hop_length=hop) #모든 주파수 빈이 각 행에 있고, 열은 시간의 흐름
    magnitude = np.abs(D) # D의 값을 (시간에 따른 각 주파수빈의 에너지 변화) 절대값 취함
    freqs = librosa.fft_frequencies(sr=sr) 
        #이게 뭔지를 모르겠네 . 이거 library 보니까 return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)를 리턴한다는데, 
        # 그럼 굳이 이걸로 쓰는 이유가 뭐야? : 이건 자동으로 n_fft를 계산해줌. 직접쓰려면 n_fft, d=1.0/sr 넣어줘야함
    avg_spectrum = np.mean(magnitude, axis=1)
        #각 주파수 빈의 배열내부 진폭값을 평균시킴

    # #피크 찾기
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(avg_spectrum,
                          height=np.max(avg_spectrum) * 0.05,
                          prominence=np.max(avg_spectrum) * 0.02,  # 돌출도 추가
                          distance=5)
        #여기서는 왜 scipy.signal 라이브버리 안에 있는걸 써야하고, find_peaks 가 전체 주파수빈에서 레벨이 큰 애들 알려주는 것. 
        #그리고 저기 peaks, _ 의 _ 자리에는 뭐가 추출돼? properties 를 추출
        # : {
            # 'peak_heights': [0.5, 0.8, 0.3],  # 각 피크의 높이
            # 'left_bases': [40, 120, 450],     # 왼쪽 베이스
            # 'right_bases': [50, 130, 460],    # 오른쪽 베이스
        # ... 등등 => 얘네와 같은 이런 dict 형태의 값들을 보여줌 (필요없으면 _로 버림)
                 #}               
        # height=np.max(avg_spectrum) * 0.1, => 최대값의 10% 이상인 피크만 찾기
        # distance=10 => 피크간 최소간격. 10칸 내에 여러 피크있으면 가장 높은것만 선택함 => index100(값 80), 인덱스 105(값 85) -> 이렇게 10개의 인덱스 내에 5칸으로 가까운 피크값이 두개가 나오면 둘중에 큰 값으로 선택
        # prominence: 주변 대비 얼마나 튀어나왔는지

    sideband_detected = False # 초기값 
    sideband_spacing = 0.0


    if len(peaks) >= 2 : 
        peak_freqs = freqs[peaks]
        spacings = np.diff(peak_freqs)

        if len(spacings) > 0:
            avg_spacing = np.mean(spacings[:3])
            spacing_std = np.std(spacings[:3])

            #간격이 일정하고 (변동 작음) 변조 주파수 범위 내
            if 1 < avg_spacing < 20 and spacing_std < 2.0:
                sideband_detected = True
                sideband_spacing = avg_spacing
                print(f"    ✅ Sideband 감지 : 변조 {avg_spacing:.2f}Hz")
            else:
                print(f"    ❌ Sideband 미감지")


    return {
    'rms': rms,
    'centroid': centroid,
    'zcr': zcr,
    'times': times,
    
    # Tremolo
    'tremolo_detected': tremolo_detected,
    'tremolo_rate': tremolo_rate,
    'tremolo_strength': tremolo_strength,

    # Vibrato
    'vibrato_detected': vibrato_detected,
    'vibrato_rate': vibrato_rate,
    'vibrato_strength': vibrato_strength,

    # 일반 AM/FM
    # 'am_detected': am_detected,
    # 'fm_detected': fm_detected,
    # 'rms_variation': rms_variation,
    # 'centroid_variation': centroid_variation,

    #사이드밴드 기반 AM/FM 추출용도
    'sideband_detected': sideband_detected,
    'sideband_spacing': sideband_spacing,
    'avg_spectrum': avg_spectrum,
    'spectrum_freqs': freqs,

    # FFT 데이터 (시각화용)
    'rms_fft': np.abs(rms_fft),
    'rms_freqs': rms_freqs,
    'centroid_fft': np.abs(centroid_fft),
    'centroid_freqs': centroid_freqs
    # 'centroid_fft' : np.abs(f0_fft),
    # 'centroid_freqs' : f0_freqs,

    }


def plot_modulation(y, sr, result, filename):
    """
    Modulation 분석 결과 시각화
    """
    
    fig, axes = plt.subplots(6, 1, figsize=(8, 8))
    
    fig.suptitle(f"Modulation analyzation - {filename}", 
                 fontsize=18, fontweight='bold')
    
    # 0. 원본 파형
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='blue')
    axes[0].set_title("waveform")
    axes[0].set_ylabel("amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 1. RMS Envelope (Tremolo)
    axes[1].plot(result['times'], result['rms'], 
                 color='red', linewidth=2, label='RMS Envelope')
    axes[1].set_title(f"RMS Envelope - Tremolo: {result['tremolo_detected']}")
    axes[1].set_ylabel("RMS")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    if result['tremolo_detected']:
        axes[1].text(0.02, 0.95, 
                     f"Tremolo: {result['tremolo_rate']:.2f} Hz\n"
                     f"intensive: {result['tremolo_strength']:.2f}",
                     transform=axes[1].transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                     fontsize=10)
    
    # 2. RMS FFT (Tremolo 주파수 분석)
    axes[2].plot(result['rms_freqs'][:50], result['rms_fft'][:50], 
                 color='darkred', linewidth=2)
    axes[2].set_title("RMS freq spectrum (Tremolo analyzation)")
    axes[2].set_ylabel("intensity")
    axes[2].set_xlabel("freq (Hz)")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0, 15])
    
    if result['tremolo_detected']:
        axes[2].axvline(result['tremolo_rate'], color='red', 
                       linestyle='--', linewidth=2, 
                       label=f"Tremolo: {result['tremolo_rate']:.2f} Hz")
        axes[2].legend()
    
    # 3. Spectral Centroid (Vibrato)
    axes[3].plot(result['times'], result['centroid'], 
                 color='green', linewidth=2, label='Spectral Centroid')
    axes[3].set_title(f"Spectral Centroid - Vibrato: {result['vibrato_detected']}")
    axes[3].set_ylabel("freq (Hz)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    
    if result['vibrato_detected']:
        axes[3].text(0.02, 0.95,
                     f"Vibrato: {result['vibrato_rate']:.2f} Hz\n"
                     f"강도: {result['vibrato_strength']:.2f}",
                     transform=axes[3].transAxes,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                     fontsize=10)
    
    # 4. Centroid FFT (Vibrato 주파수 분석)
    axes[4].plot(result['centroid_freqs'][:50], result['centroid_fft'][:50], 
                 color='darkgreen', linewidth=2)
    axes[4].set_title("Centroid freq detection (Vibrato detection)")
    axes[4].set_ylabel("intensity")
    axes[4].set_xlabel("freq (Hz)")
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim([0, 15])
    
    if result['vibrato_detected']:
        axes[4].axvline(result['vibrato_rate'], color='green', 
                       linestyle='--', linewidth=2, 
                       label=f"Vibrato: {result['vibrato_rate']:.2f} Hz")
        axes[4].legend()
    
    # 5. Zero Crossing Rate
    axes[5].plot(result['times'], result['zcr'], 
                 color='purple', linewidth=2, label='ZCR')
    axes[5].set_title("Zero Crossing Rate")
    axes[5].set_ylabel("ZCR")
    axes[5].set_xlabel("time (s)")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend()
    
    plt.tight_layout()
    
    name_only = os.path.splitext(filename)[0]
    plt.savefig(f"modulation_{name_only}.png", dpi=150)
    # plt.show()



def main():
    AUDIO_FOLDER = "Librosa-basics/audio_sample_modulation"
    audio_files = sorted(glob.glob(f"{AUDIO_FOLDER}/*.wav"))
    
    if len(audio_files) == 0:
        print(f"'{AUDIO_FOLDER}'에 .wav 파일이 없습니다.")
        print(f"현재 위치: {os.getcwd()}")
        return
    
    print("=" * 80)
    print(f"📁 폴더: {AUDIO_FOLDER}")
    print(f"📊 발견된 파일: {len(audio_files)}개")
    print("=" * 80)
    
    for path in audio_files[:1]:
        filename = os.path.basename(path)
        
        print(f"\n{'─'*60}")
        print(f"📄 {filename}")
        
        try:
            y, sr = librosa.load(path, mono=True, sr=None)
            
            print(f"  샘플레이트: {sr} Hz")
            print(f"  길이: {len(y)/sr:.2f}초")
            
            result = detect_modulation(y, sr)
            plot_modulation(y, sr, result, filename)
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✅ 모든 파일 처리 완료!")


if __name__ == "__main__":
    main()



"""rms_freqs = np.fft.rfftfreq(len(rms), d=(hop/sr))  

주파수 간격 (rms_freqs = [0, Δf, 2Δf, 3Δf, ..., Nyquist])
    Δf = 1 / (n × d)
    = 1 / (432 × 0.0116)
    = 1 / 5.01
    = 0.2 Hz
"""


""" FM 감지 vs Vibrato 감지

(1) FM detection
+ 방법 : std(diff(centroid))
 

(2) Vibrato detection
+ 방법 : FFT + 피크 찾기


=> if, 둘다 주파수의 변화로만 감지하면 

    예시 1: 불규칙한 변화
    centroid = [1000, 3000, 500, 4000, 800]
    diff = [2000, -2500, 3500, -3200]
    std(diff) = 2600 Hz → ✅ FM 감지!

    예시 2: 주기적 변화 (Vibrato)
    centroid = [1000, 1200, 1000, 1200, 1000]
    diff = [200, -200, 200, -200]
    std(diff) = 200 Hz → ✅ FM 감지!

    → 둘 다 FM으로 감지해버린다


    => vibrato 는 신호를 fft 로 주파수 도메인(시간 -> 주파수)으로 변환하는데
        1. 시간 도메인 RMS 
            rms = [0.5, 0.8, 0.5, 0.8, 0.5, 0.8, ...]  # 시간 흐름에 따른 각 freq bin 의 magnitude를 보여줌
            times = [0, 0.0116, 0.0232, 0.0348, ...]    # 각 프레임의 시간을 보여줌

        2. FFT를 통한 주파수 도메인 변환
            rms_freqs = [0, 0.46, 0.92, 1.38, 1.84, 2.3, ...]  Hz => 시간정보는 사라지고, 전체신호에 각 주파수 대역에 얼마나 들어있나만 알려줌
            rms_power = [10,  5,   200,   8,    7,   6,  ...]  각 주파수 대역의 강도 # 총 95개 (188/2 + 1)
                                    ↑(피크)

            => 시간 정보가 사라지기 때문에 전체 오디오 길이동안 어느 주파수 성분이 강했는지 알수있음
            => rms_power 의 정보에서 가장 강하게 나오는 index를 보면, 전체 음원 통틀어 어떤 주파수에서 제일 레벨이 큰지 알 수 있는데,
            
        **FFT는 항상 “무엇이 반복되고 있는가”를 찾는다
            audio → 공기 진동이 반복됨 (audio FFT(ex. Spectrum) : 소리의 구성 주파수를 보여줌)
            RMS → loudness가 반복됨 (RMS FFT(ex. Tremolo rate 검출 : 소리의 변화 속도(변조 주파수)를 보여줌)


"""


"""np.fft.rfft vs np.fft.rfftffreq 

    # FFT 로 변조 주파수 찾기
    centroid_fft = np.fft.rfft(centroid_normalized)
    centroid_freqs = np.fft.rfftfreq(len(centroid), d=(hop/sr))
    centroid_power = np.abs(centroid_fft)

    (1)np.fft.rfft(signal)
     : 실수 신호를 입력받아 FFT 계산을 수행 -> 복소수 스펙트럼을 반환

    (2)np.fft.rfftfreq(n, d) 
     : rfft 결과에 대응하는 주파수 값들을 생성
        - n : 원본 신호 길이
        - d : 샘플 간격( 1 / sample_rate ) => rfft 가 반환한 각 bin 이 실제로 몇 Hz인지 알려주는 배열

        
"""

""" tremolo 와 fft ...

(1) 내가 알고있던 EQ 에서의 스펙트럼 
 : FFT 에 입력하는것이 오디오파형임. -> 그래서 freq domain 에서 보는건 소피의 주파수

(2) tremolo 의 분석에서의 FFT 는
 : RMS (변화)값 이 입력임 -> 그래서 FFT 의 결과로 보는건 변화 속도. 6Hz 와 같은 레이트

 ++FFT : 반복 속도 분석기 . 무엇을 FFT 하느냐에 따라 그 주파수의 의미가 완전히 달라짐!

** tremolo 있는 신호 : audio(t) = sin(2π·440t) * (1 + sin(2π·6t)) 
    # + 1 은 원래 존재하는 기본 음(캐리어)를 유지하기 위한 DC 바이어스
    # 이걸 안하면 carrier 가 사라지고, sideband 만 남는 이상한 구조가 됨  => RM 


**EQ 에서는 (Audio FFT) : 이걸 여러개의 고정된 주파수로 쪼개자 !
    => 결과 
    434Hz (440 - 6)
    440Hz
    446Hz (440 + 6)

"""


"""RM 과 AM 

(1)RM
x(t) = sin(2π·440t) * sin(2π·6t)

sin(A)·sin(B)
= 0.5[cos(A-B) - cos(A+B)]
= 0.5[cos(440-6)t - cos(440+6)t]


(2)AM 
x(t) = sin(2π·440t) * (1 + sin(2π·6t))

"""


"""FM의 사이드밴드 공식 (베셀 함수)

python# FM 변조 지수 (β)
β = frequency_deviation / modulator_freq

# 예시 1: β = 0.5 (작은 변조)
440Hz: 주성분
440±6Hz: 약한 사이드밴드
→ 사이드밴드 간격 일정함 ✅

# 예시 2: β = 5 (큰 변조)
440Hz: 약함
440±6Hz: 강함
440±12Hz: 강함
440±18Hz: 중간
440±24Hz: 약함
→ 간격은 6Hz로 일정하지만, 
   강도가 불규칙 → 피크 선택 어려움 ❌
"""
