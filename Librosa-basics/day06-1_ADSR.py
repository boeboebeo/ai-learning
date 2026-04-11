#day 06-1 ADSR extraction

import glob #폴더에 있는 모든 오디오 파일 자동처리
import os
import librosa 
import numpy as np
import matplotlib.pyplot as plt
import csv

#한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


# ══════════════════════════════════════════════════════════════
# 함수: ADSR 추출
# ══════════════════════════════════════════════════════════════
def extract_adsr(y, sr, hop=512):

    # 1. RMS 계산
    # RMS = Root Mean Square (에너지 측정)
    # hop_length마다 계산 -> 시간 해상도 결정
    rms = librosa.feature.rms(y=y, hop_length=hop, center=True)[0]
        #우선 librosa.feature.rms 가 어떻게 출력되는지를 알아야 함 
        # => 배열 형태로 출력. 전체 샘플의 개수를 hop 으로 나눈만큼의 index개수를 가진 배열로 출력된다.
        #내부 padding = > 처리 되어있음 (아래 주석 참고)

    # 디버깅용 print
    # print(rms) # rms 
    # print(f"\nrms 배열 개수 구하는 법 : {len(y)/hop}") # y = 전체 샘플(96000) , hop=512로 나눔 = 96000 / 512 = 187.5
    # print(len(rms)) #282출력됨 그렇다면 전체시간을 512로 나눴을때 이렇게 처리되고 있는건가?
    # duration_sec = len(y) / sr #전체 샘플의 개수를 샘플레이트로 나누면 이 파일의 총 시간이 나옴
    # print(f"duration : {duration_sec * 1000} ms")
    # print(f"max rms : {np.argmax(rms)}") #2 -> 세번째 인덱스의 출력이 제일 높게 나옴


    #각 RMS 프레임의 시간 (초 단위) ?
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
        # times[0] = 첫번째 프레임의 중심시간 => 따라서 출력해보면 index 0에 해당하는 시간이 0임을 알 수 있음
        # but, 
        # 각 rms 배열의 index에 해당하는 타임을 배열형태로 출력함

    # print(f"times : {times[0]:.10f}") 
        #[0.         0.01066667 0.02133333 0.032      0.04266667 0.05333333 
        # 위와 같이 시간에 대한 정보를 배열 형태로 알려줌 (numpy auto format 때문에 소수점 뒤 자리수가 달라짐. 가독성 향상측면)


    # 2. RMS - 최대값 계산 (근데 여기서는 PPM의 peak가 아니고, 제일 높은 레벨의 RMS 값)
    rms_peak_idx = np.argmax(rms) #RMS 최댓값 인덱스
    rms_peak_val = rms[rms_peak_idx] #RMS 최댓값 레벨

    # print(times[2]) 
        # 최대 rms 의 times 프레임이 index 3이었는데, 그래서 attack이 21.3ms가 나왔음
        # peak 로 바꿔보자 (RMS의 시간 해상도의 한계가 있음. padding도 되어있음 심지어)
    

    # 2-1 PEAK 계산 - 진짜 PEAK LEVEL 계산 
    peak_peak_idx = np.argmax(np.abs(y))
    peak_attack_ms = (peak_peak_idx / sr) * 1000
    


    # 3. Attack 계산
    # 0초부터 Peak 까지의 계산
    attack_ms = times[rms_peak_idx] * 1000 # sec -> ms

    # 4. Decay 끝 찾기 
    # Peak 이후 기울기가 거의 0이 되는 지점
    decay_end = rms_peak_idx
    
    # Peak 이후 부분만 검사
    after_peak = rms[rms_peak_idx:] 

    # 기울기가 거의 평평해지는 지점 찾기
    for i in range(10, min(len(after_peak) - 10, 200)): 
            # 10부터, after_peak 의 idx의 개수에서 10뺀것과, 200 중의 작은 것 고르기.
            # 아마 전체 인덱스 넘어갈까봐 이렇게 하는것 같긴한데 , 근데 그럼 왜 200이지? 
        #최근 10프레임 기울기 
        recent_slope = after_peak[i] - after_peak[i-10]
            # 피크 이후의 10번째 인덱스의 rms - 피크의 인덱스 시작이 10이므로 -10 하면 그냥 after_peak[0] 이여서 피크 자기자신의 RMS값
            # 그럼 우선 피크 바로 직후에는 after_peak[i] 가 더 작을테니 음수가 나옴 -> 그래서 절대값 abs . 근데 점점 위 인덱스로 갈수록 
            # after_peak[i]와 after_peak[i-10]이 별로 레벨 차이가 안나게 되면서 0에 가까운 수치가 된다. 

        # 기울기 < 1% -> decay 끝!
        if abs(recent_slope) < rms_peak_val * 0.01: #왜 또 갑자기 peak_val 에 0.01을 곱하는거지?
            decay_end = rms_peak_idx + i #암튼 그렇게 별차이 안나는 그 지점을 decay가 끝나는 지점으로 확인
            break

    # 5. sustain 레벨 찾기
    # Decay 끝 이후 구간의 중앙값 
    sustain_start = min(decay_end + 5, len(rms) - 30) 
        # decay_end 의 인덱스에 +5 한 곳, 과 rms 전체의 프레임의 개수에서 30뺀 것 중 작은거 고름
        # 이게 또 뭔 기준일까...
    sustain_end = min(sustain_start + 50, len(rms))

    if sustain_end > sustain_start:
        sustain_val = np.median(rms[sustain_start:sustain_end]) #중간값 찾아서 sustain_val 로 지정
    else:
        #너무 짧은 소리의 경우, 인덱스 모자람
        sustain_val = rms[-1] if len(rms) > 0 else rms_peak_val * 0.1


    # 6. Decay 시간 계산
    decay_ms = (times[decay_end] - times[rms_peak_idx]) * 1000 # ms변환 하기 위해 *1000함

    # 7. Sustain 레벨 (dB)
    # peak 대비 상대적 레벨
    sustain_db = 20 * np.log10(sustain_val / (rms_peak_val + 1e-8)+ 1e-8)


    # 8. Release 시작점 찾기
    # 뒤에서부터 threshold 이상인 마지막 지점 
    threshold = rms_peak_val * 0.01 #Peak 1%
    release_start = len(rms) - 1

    for i in range(len(rms)-1, decay_end, -1): #이건 아마도 뒤에서 부터 판단하는걸까?
        if rms[i] > threshold:
            release_start = i
            break

    # 9. Release 시간 계산
    release_ms = (times[-1] - times[release_start])*1000 
        # 시간의 맨 뒤의 값 (전체 시간) - release start 빼고 ms 로 표현하기 위해서 1000 곱합.
        # 근데 전체시간의 뒤의 갑셍서 빼면 안되고, 볼륨이 0이 된 시점까지 계산해야 하는거 아닌가?

    return {
        'attacks_ms' : attack_ms,
        'peak_attack_ms' : peak_attack_ms,
        'decay_ms' : decay_ms,
        'sustain_db' : sustain_db,
        'release_ms' : release_ms,
        'peak_idx' : rms_peak_idx,
        'decay_end' : decay_end,
        'sustain_start' : sustain_start,
        'release_start' : release_start

    }


# ══════════════════════════════════════════════════════════════
# 메인: 배치 처리
# ══════════════════════════════════════════════════════════════
def main():
    #폴더의 모든 .wav 파일 분석

    # 설정
    AUDIO_FOLDER = "Librosa-basics/audio_sample_ADSR"
    
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
            y, sr = librosa.load(path, sr=None)

            print(f"    샘플레이트 : {sr} Hz")
            print(f"    길이 : {len(y)} samples ({len(y)/sr:.2f}초)")

            # ADSR 추출
            adsr = extract_adsr(y, sr, hop=512)

            print(f"\n ======== ADSR ========")
            print(f"    RMS_Attack  : {adsr['attacks_ms']:6.1f}ms")
            print(f"    Peak_Attack : {adsr['peak_attack_ms']:6.1f}ms")
            print(f"    Decay   : {adsr['decay_ms']:6.1f}ms")
            print(f"    Sustain : {adsr['sustain_db']:6.1f}dB")
            print(f"    Release : {adsr['release_ms']:6.1f}ms")            

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






#print(f"현재위치 : {os.getcwd()}") #현재 파일경로 찾는법 (current working directory)

# audio_files = glob.glob("Librosa-basics/audio_sample_ADSR/*.wav")

# #print(len(audio_files))

# for path in audio_files:
#     filename = os.path.basename(path) #파일 이름만 추출
#     print(f"Processing : {filename}")


"""
hop vs n_fft
=> 각각 독립적으로 소리 신호를 나눈다.

1)n_fft : 주파수 축 나누기 ()
    근데 입력(시간영역. samples) 2048개의 샘플 => FFT => 출력(주파수 영역. hz) 의 1025개의 주파수 bins 을 만들어내는 것.
    n_fft = 4096 이면, 출력 bins = 2049개, 주파수 간격 sr / 4096 으로 더 좁음 ( 더 높은 주파수 해상도 )

    FFT = sample -> Hz 변환기
    audio bin 의 개수 = n_fft // 2 + 1 = 4096/2 + 1 = 2049개의 오디오 빈이 생성됨

2)hop : 시간 축 나누기 (샘플단위 -> 그래서 계산 후 ms 로 바꾸기 위해서 *1000 함)

"""

""" rms 의 padding 

times[0], times[1]과 같은 값을 출력할때 중심 시간을 기준으로 시간을 알려주는데, 

center = False 일 경우 (기본값은 True)
times[0] 과 같은 이 첫번째 프레임이나 끝 프레임에서는 중심 값을 시작으로 해버리면 중심이 = 샘플 1024가 되기 때문에 시작이 아니게 됨
따라서, 패딩을 1024개의 샘플을 (frame_length = 기본값 2048 의 절반)을 앞 뒤로 배치하여 
첫번째 프레임의 시간이 0일때가 중간이 되게 함 

패딩 추가:
[패딩1024] + [샘플0, 샘플1, ..., 샘플2047, ...] + [패딩1024]

Frame 0: [패딩512 ~ 패딩1024, 샘플0 ~ 샘플1535]
                              ↑
                           중심 = 샘플0 ✅

중심 시간 = 0 / sr = 0.000초 ✅

→ times[0] = 0.000초 (정확한 시작!)


"""

"""Attack 추출 -> RMS , PEAK

RMS로 Attack 타임을 구했더니 Attack 이 0인 파일도 자꾸 어택타임이 RMS 세번째 인덱스(레벨 최대 지점)의
시간으로 나와서 -> 21.3ms (print(times[2])요 지점이 최대로 나오고, 여기가 0.0213s 정도 나옴)

PEAK로 Attack 타임을 구해봤다. 그래도 0은 안나오지만 훨씬 가까워짐
위의 peak_peak_idx : 가 진짜 peak 레벨 기준으로 구한 것 .

"""