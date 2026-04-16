# day 08 - 1 modulation_estimation 

import librosa
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import platform


def detect_modulation(y, sr, hop=512):
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

    #FFT 로 변조 주파수 찾기
    rms_fft = np.fft.rfft(rms_normalized)
        #rfft = Real FFT (실수 : Real 입력 전용). 출력은 N개에서 N개 복소수
        #fft = input에 복소수 가능함. 출력은 N개에서 N/2 + 1 개 복소수가 출력됨 (실수신호는 주파수가 대칭이라서. Nyquist 주파수 까지만)
    rms_freqs = np.fft.rfftfreq(len(rms), d=(hop/sr)) 
        #각 fft 결과가 몇 Hz인지
        #(입력길이, d=샘플 간격) . d = hop/sr = 512/44100 = 0.0116초 => 주파수 배열 생성
    rms_power = np.abs(rms_fft)

    print(len(rms_power))

    #DC 제외하고 가장 강한 주파수
    if len(rms_power) > 1: 
        #이건 당연히 1보다 큰거 아닌가? 지금 rms_power 배열의 개수가 95개인데... 근데 왜 95개지?
        peak_idx = np.argmax(rms_power[1:20]) + 1 # 1~20Hz 범위에서 
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


    #------Vibrato detection------
    print(f"\n  [Vibrato Analyze]")

    #spectral Centroid 의 주기적 변화
    centroid_normalized = (centroid - np.mean(centroid)/(np.std(centroid)+ 1e-8))

    # FFT 로 변조 주파수 찾기
    centroid_fft = np.fft.rfft(centroid_normalized)
    centroid_freqs = np.fft.rfftfreq(len(centroid), d=(hop/sr))
    centroid_power = np.abs(centroid_fft)

    #DC 제외하고 가장 강한 주파수
    if len(centroid_power) > 1:
        peak_idx = np.argmax(centroid_power[1:15]) + 1 # 1 ~ 15Hz 범위에서 
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


    # ── 일반 AM/FM 감지 ──
    print(f"\n  [Modulation Analyze]")
    
    # AM (진폭 변조)
    rms_variation = np.std(np.diff(rms))
    if rms_variation > 0.01:
        am_detected = True
        print(f"    ✅ AM 감지: 변화량 {rms_variation:.4f}")
    else:
        am_detected = False
        print(f"    ❌ AM 미감지")
    
    # FM (주파수 변조)
    centroid_variation = np.std(np.diff(centroid))
    if centroid_variation > 100:
        fm_detected = True
        print(f"    ✅ FM 감지: 변화량 {centroid_variation:.1f} Hz")
    else:
        fm_detected = False
        print(f"    ❌ FM 미감지")  


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
    'am_detected': am_detected,
    'fm_detected': fm_detected,
    'rms_variation': rms_variation,
    'centroid_variation': centroid_variation,

    # FFT 데이터 (시각화용)
    'rms_fft': np.abs(rms_fft),
    'rms_freqs': rms_freqs,
    'centroid_fft': np.abs(centroid_fft),
    'centroid_freqs': centroid_freqs

    }



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
            # plot_modulation(y, sr, result, filename)
            
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

