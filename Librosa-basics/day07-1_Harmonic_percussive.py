# day07 -1 harmonic vs percussive classification

import librosa
import numpy as np
import glob #폴더에 있는 모든 오디오 파일 자동처리
import os
import matplotlib.pyplot as plt



def HarmonicPercClassification(y, sr, filename, hop=512):

    print(f"\n [HPSS 분석중...]")
    # ── 2. Harmonic / Percussive 분리 ─────────────────────────────
    
    # HPSS : 멜로디 성분 vs 타악 성분 분리
    y_h, y_p = librosa.effects.hpss(y, margin=10.0)
        #hamronic, percussive 로 구분
        #그래서 각각의 y_h, y_p에는 시간에 따른 진폭값 만 남아있음

    # 에너지 계산
    h_energy = np.sum(y_h ** 2)
    p_energy = np.sum(y_p ** 2)
    total = h_energy + p_energy
    
    h_ratio = h_energy / total
    p_ratio = p_energy / total
    
    print(f"  Harmonic:   {h_ratio:.1%}")
    print(f"  Percussive: {p_ratio:.1%}")


    # 스펙트로그램
    D_original = librosa.stft(y)
    D_harmonic = librosa.stft(y_h)
    D_percussive = librosa.stft(y_p)



    S_o = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
    S_h = librosa.amplitude_to_db(np.abs(D_harmonic), ref=np.max)
    S_p = librosa.amplitude_to_db(np.abs(D_percussive), ref=np.max)
        

    # 스펙트로그램 색상 범위 통일 
    vmin = min(np.min(S_h), np.min(S_p))
    vmax = max(np.max(S_h), np.max(S_p))
    

    #그래프 
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

    # ✅ 전체 제목 (맨 위 큰 글씨)
    fig.suptitle(f"HPSS Analyzation - {filename}", 
                 fontsize=18, 
                 fontweight='bold')

    # 원본의 최대/최소값 구하기
    y_max = np.max(np.abs(y)) + 0.1  # 절댓값의 최대
    y_min = -y_max  # 대칭

    #0행 원본
    librosa.display.waveshow(y, sr=sr, ax=axes[0, 0], color="#888888")
    axes[0, 0].set_title("original waveform")   
    axes[0, 0].set_ylim([y_min, y_max])  # ✅ y축 최대값을 동일하게 하여 객관적으로 비교


    librosa.display.specshow(S_o, sr=sr, vmin = vmin, vmax = vmax,
                            ax=axes[0, 1])
    axes[0, 1].set_title("original spectrogram")


    #1행 : harmonic 
    librosa.display.waveshow(y_h, sr=sr, ax=axes[1, 0], color="#4A90D9")
    axes[1, 0].set_title("Harmonic waveform")
    axes[1, 0].set_ylim([y_min, y_max])  # ✅

    librosa.display.specshow(S_h, sr=sr, vmin = vmin, vmax = vmax,
                         ax=axes[1, 1])
    axes[1, 1].set_title("Harmonic spectrogram")   

    # 2행: Percussive
    librosa.display.waveshow(y_p, sr=sr, ax=axes[2, 0], color="#E85D8A")
    axes[2, 0].set_title("Percussive waveform")
    axes[2, 0].set_xlabel("time(s)")
    axes[2, 0].set_ylim([y_min, y_max])  # ✅

    librosa.display.specshow(S_p, sr=sr, vmin = vmin, vmax = vmax,
                            ax=axes[2, 1])
    axes[2, 1].set_title("Percussive spectrogram")
    axes[2, 1].set_xlabel("time(s)")



    plt.tight_layout()
    plt.show()

    # 에너지 분석 : 타악 비율이 높으면 → 노이즈 오실레이터, 드럼 샘플
    harmonic_energy   = np.mean(librosa.feature.rms(y=y_h))
    percussive_energy = np.mean(librosa.feature.rms(y=y_p))
    perc_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-8)
        #퍼커션의 비중이 더 높은지, harmonic 의 더 비중이 높은지 비교



    print(f"\n  [에너지 분석]")
    print(f"    Harmonic 에너지 : {harmonic_energy:.6f}")
    print(f"    Percussive 에너지 : {percussive_energy:.6f}")
    print(f"    Percussive 비율 : {perc_ratio:.2%}") # % ? 

    if perc_ratio > 0.5:
        print(" -> 노이즈 성분 강함")
    elif perc_ratio > 0.2:
        print(" -> 노이즈 약간 있음")
    else:
        print(" -> 신스/악기 사운드")



# ══════════════════════════════════════════════════════════════
# 메인: 배치 처리
# ══════════════════════════════════════════════════════════════
def main():
    #폴더의 모든 .wav 파일 분석

    # 설정
    AUDIO_FOLDER = "Librosa-basics/audio_sample_percvswave"
    
    # 파일 찾기
    audio_files = sorted(glob.glob(f"{AUDIO_FOLDER}/*.wav")) #그냥 출력하면 사실 무작위로 리스트가 작성되기 때문에 sorted 넣어주기 -> 알파벳순으로 정렬가능

    if len(audio_files) == 0 :
        print(f"'{AUDIO_FOLDER}'에 .wav 파일이 없습니다.")
        print(f"현재 위치 : {os.getcwd()}")
        return 
    
    

    print("=" * 80)
    print(f"📁 폴더: {AUDIO_FOLDER}")
    print(f"📊 발견된 파일: {len(audio_files)}개")
    print("=" * 80)

    # 결과 저장용 리스트
    results = []

    # 각 파일 처리
    for path in audio_files: #첫번째만 시험해보려 할때는 audio_files[:1] 붙이기
        filename = os.path.basename(path)
        name_only = os.path.splitext(filename)[0]

        print(f"\n{filename}")

        try:
            # 오디오 로드
            # sr=None: 원본 샘플레이트 유지
            y_raw, sr = librosa.load(path, mono=False, sr=None) #입력 신호를 모노로 받기 때문에 레벨에 문제가 생겼어서, 스테레오 받은 후 l ,r 비교해서 많이 같으면 l 만 쓰기로! 

            if y_raw.ndim == 1:
                # 입력소스가 mono라면 -> 그대로 
                y = y_raw

            else:
                # Stereo 
                left = y_raw[0]
                right = y_raw[1]

                # 상관계수
                correlation = np.corrcoef(left, right)[0, 1]

                if correlation > 0.99:
                    #거의 같음 
                    print(" ℹ️ Stereo 이지만 L/R 동일 (Left 사용)")
                    y = left
                
                else:
                    # L, R 거의 다름 -> 에너지 합산
                    print(" ℹ️ Stereo 다름 (에너지 합산)")
                    y = np.sqrt(left**2 + right**2) # return 대신 y에 이거 들여보냄


            print(f"    샘플레이트 : {sr} Hz")
            print(f"    길이 : {len(y)} samples ({len(y)/sr:.2f}초)")

            HarmonicPercClassification(y, sr, filename)



        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("-" * 80)

# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()


"""HPSS
: 한 오디오를 Harmonic & Percussive 성분으로 분리
    1. Harmonic : 음정이 있는 소리, 멜로디/코드, 지속되는 톤
    2. Percussive : 음정이 없는 소리, 리듬/타격음, 짧고 순간적

*도출 과정*
1단계. STFT
오디오 -> 스펙트로그램

2단계. 패턴분석
Harmonic : 주파수 축으로 연속(가로로 긴 줄)
Percussive : 시간축으로 짧음 (세로로 짧은 막대)

3단계. 필터링
각각 분리해서 다시 오디오로 변환


=> Decompose an audio time series into harmonic and percussive components.
    This function automates the STFT->HPSS->ISTFT pipeline, and ensures that
    the output waveforms have equal length to the input waveform ``y``.

+istft (Inverse STFT) : 
    : 역변환. 
    : stft 는 시간 도메인 -> 주파수 도메인으로 변환하는 것 (오디오(샘플들) -> 스펙트로그램)
    : istft 는 주파수 도메인 -> 시간도메인으로 변환하는 것 (스펙트로그램(주파수*시간) -> 오디오)
    

+hpss 함수 설명에서 
    y_perc = core.istft(
        )  => 의 core. 는 librosa내부의 core 라는 핵심기능들 안에서 istft 를 함수를 쓰겠다고 가지고 오는 것

*decompose = 분해모듈
librosa.decompose. ____ : 여기 ____에 들어가는게 decompose의 주요 함수들 

    주요 함수:
    - hpss()   # Harmonic-Percussive 분리
    - nn_filter()  # Non-negative filtering
    - decompose()  # 일반 분해
            
"""