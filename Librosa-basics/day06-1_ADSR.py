#day 06-1 ADSR extraction

    #ADSR 추출 코드 완전 정리! 🎵 이거 검색해서 참조

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
        # => 배열 형태로 출력. 전체 샘플의 개수를 hop 으로 나눈만큼의 index개수 + 1 을 가진 배열로 출력된다. => 올림+패딩 효과때문에
        #내부 padding = > 처리 되어있음 (아래 주석 참고)

    # 디버깅용 print
    # print(rms) # rms 
    # print(f"\nrms 배열 개수 구하는 법 : {len(y)/hop}") # y = 전체 샘플(96000) , hop=512로 나눔 = 96000 / 512 = 187.5
    # print(len(rms)) #282출력됨 그렇다면 전체 샘플을 512로 나눴을때 이렇게 처리되고 있는건가?
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
        # 얘가 문제다. 왜 decay_end = rms_peak_idx 로 되어있는거지?
        # 아 우선 초기값 이였고, 만약 밑에서 decay_end 값을 못찾으면 이 peak 값이 그대로 decay_end 값으로 유지함
    
    # Peak 이후 부분만 검사
    after_peak = rms[rms_peak_idx:]
    threshold = rms_peak_val * 0.01 #Peak 1% (피크에 레벨에 따라서 다른 threshold를 만들게 함)


    # 모든 기울기 계산 => 아래의 if 문을 통해서 기울기가 rms_peak_val * 0.01 한것을 못 넘는것 같아서 기울기 계산 해본것!
    # slopes = [abs(after_peak[i] - after_peak[i-10]) 
    #       for i in range(10, min(len(after_peak)-10, 200))]

    # # 최소값 확인
    # if slopes and min(slopes) >= threshold:
    #     print(f"⚠️ 경고: 모든 기울기가 threshold({threshold:.6f}) 이상!")
    #     print(f"        최소 기울기: {min(slopes):.6f}")
    #     print(f"        → Decay 못 찾을 거예요")
        #현재 threshold 는 0.000392인데 모든 기울기가 그것보다 다 큼 (0.000555가 최소값임)


    # (Decay 찾기!) 1단계 : 기울기로 찾기
    decay_found = False 
        #decay 찾았는지 여부, 1-2-3단계에 거쳐서 찾게 되는데, 각 단계의 시작단계에서 decay_found = True 가 되어있으면 이미 찾은것이므로 종료하게 됨

    # 기울기가 거의 평평해지는 지점 찾기
    for i in range(10, min(len(after_peak) - 10, 200)): 
            # 10부터, after_peak 의 idx의 개수에서 10뺀것과, 200 중의 작은 것 고르기.
            # 아마 전체 인덱스 넘어갈까봐 이렇게 하는것 같긴한데 , 근데 그럼 왜 200이지? 
                # 200 프레임 이라면 200 * 512 / 22050 = 약 4.6초 정도됨 
                # 4.6초 이상이기 때문에 Decay 값 비현실적 => 너무 긴 구간은 검색 안해도 되니까 경험적 상한선으로 200으로 정해둠

        #최근 10프레임 기울기 
        recent_slope = after_peak[i] - after_peak[i-5]
            # 피크 이후의 10번째 인덱스의 rms - 피크의 인덱스 시작이 10이므로 -10 하면 그냥 after_peak[0] 이여서 피크 자기자신의 RMS값
            # 그럼 우선 피크 바로 직후에는 after_peak[i] 가 더 작을테니 음수가 나옴 -> 그래서 절대값 abs . 근데 점점 위 인덱스로 갈수록 
            # after_peak[i]와 after_peak[i-10]이 별로 레벨 차이가 안나게 되면서 0에 가까운 수치가 된다. 


        # 기울기 < 1% -> decay 끝!
        if abs(recent_slope) < rms_peak_val * 0.01: 
                #왜 또 갑자기 peak_val 에 0.01을 곱하는거지?
                # => rms_peak_val * 0.01 => Peak 의 1% : 절대 기울기가 아니라 상대 기울기
                # 피크가 크면 같은 1%의 변화도 큰 값이 됨. (peak 크기에 비례하는 임계값으로 구하려고)
            decay_end = rms_peak_idx + i # 암튼 그렇게 별차이 안나는 그 지점을 decay가 끝나는 지점으로 확인
            decay_found = True
            break
            # 만약 decay 를 못찾으면 그냥 decay_end = peak 값을 그대로 유지함
            # 근데 지금은 아마도 계속 그 값이 true가 되는 순간이 안오기 때문에 break 가 안되는듯!! 그래서 decay_end 가 처음의 그 peak값을 유지함
            # 위의 우선 모든 기울기를 계산해서 전체의 기울기가 rms_peak_val * 0.01 보다 작아지는 순간이 오는지 확인 

    # print(decay_found) #지금 False 가 출력이 되므로 

    # (Decay 찾기) 2단계 : 못찾았으면 레벨로 찾기 

    if not decay_found:
        level_threshold = rms_peak_val * 0.3 # 피크의 30%의 볼륨

        for i in range(len(after_peak)):
            if after_peak[i] < level_threshold: #피크이후의 어떤 rms 레벨이 level_threshold(peak_val * 0.3 한 레벨)을 넘어가게 되면 그걸 decay_end 로 지정함
                decay_end = rms_peak_idx + i # 넘어간다면, rms_peak_idx + i(넘어갔을때의 그 인덱스) 한 그 만큼을 decay_end 로 지정
                decay_found = True #그리고 decay_found 는 True 로 처리해서 그 아래의 계산단계는 처리 안되게 함

    # (Decay 찾기) 3단계 : 그래도 못 찾으면 나중에 처리
    if not decay_found:
        print(f"⚠️ Decay : 끝까지 감소 (Release 후 재계산)")



    # (개선) 4-2. release 시작 먼저 찾기 
    release_start = len(rms) - 1 
        # 우선 release_start값을 마지막 인덱스 값으로 두면서, 밑의 조건에 맞는 값으로 release start 지점을 바꿈
        # 전체 오디오파일 끝날때까지 노트가 안끝날수도 있으니까

    for i in range(len(rms)-1, decay_end, -1): # 뒤에서 부터 셈
        if rms[i] > threshold: #여기서 threshold 는 전체 피크의 0.01되는 지점
            release_start = i  #각 인덱스의 레벨이 threshold 보다 커지는 순간, release start 는 i index가 되고, for문은 중단됨
            break

        # rms 인덱스의 전체 개수부터, peak 값 까지 뒤에서 부터 순서대로 구하면서 
    # print(f"\nthreshold : {threshold}")

    # 만약 release 를 못 찾았으면 (모두 threshold 이하라면) (그럴 수 있나?)
    if release_start == len(rms) - 1:
        #전체가 threshold 이하 -> release 없음
        release_start = min(decay_end + 10, len(rms) -1 )


    # 5. sustain 레벨 찾기 (개선)
    # Decay 끝 이후 구간의 중앙값 
    # decay_end_safe = decay_end + 5 #decay 끝에 +5를 더한 인덱스를 decay_end_safe 변수로 지정
    # release_start_safe = release_start - 5 #release_start 지점에서 -5를 한 지점도 release_start_safe 로 지정
        

    
        #decay 끝과, release 시작 사이의 공간을 두고 만약 그게 최소 10프레임을 가진다면 sustain 길이가 있는것.
    # available_frames = release_start_safe - decay_end_safe

    # if available_frames >= 10:
    #     sustain_start = decay_end_safe #릴리즈 시작-디케이 끝 한 지점 사이의 공간이 10프레임 이상이면 그냥 sustain_start 지점과 decay_end_safe 지점을 같게 하기
    #     sustain_end = min(sustain_start + 50, release_start_safe)
    #         # sustain_end => 서스테인 시작지점에서 50더한것과 릴리즈 스타트 지점 중 작은걸로 end 지정하기 
    #         # sustain_start + 50 한 지점이 전체 길이에서 벗어날 수 도 있으니 (릴리즈 구역 침범할 수 도 있으니)

    #     sustain_val = np.median(rms[sustain_start:sustain_end])
    #         # 근데 중간값을 골라야 하는게 아니고, 어짜피 sustain 레벨 안이라면 그냥 다 같아야 하는거 아닌가?
    #     sustain_present = True 
    #         # 서스테인 존재 

    # else : 
    #     # 서스테인이 없다면
    #         sustain_val = 0
    #         sustain_present = False
    #         print(f"⚠️ Sustain 구간 부족 ({available_frames}Frames)")


    # 6. Sustain 구간 설정 (개선)


    sustain_start = min(decay_end + 5, len(rms) - 30) 
        # decay_end 의 인덱스에 +5 한 곳, 과 rms 전체의 프레임의 개수에서 30뺀 것 중 작은거 고름
        # decay_end + 5 : decay 가 끝나고 5프레임의 여유
        # len(rms) - 30 : 끝에서 30프레임 전 (Release 구간을 침범하는걸 방지하기 위해서)
        # => 둘 중 짧은것을 고름. 짧은 소리에 대비하기 위해 
    sustain_end = release_start - 5
        # decay_end 지점과, sustain_start 지점은 적어도 5 프레임 정도의 차이를 가질 수 밖에 없음
        # 또한, sustain_end 지점은 최소가 start + 50 인 지점이므로, 그 안에 노트의 길이가 끝난다면 사실 정확한 sustain 의 끝 지점을 구할 수 없음
        # => 따라서 release를 먼저 찾게끔 코드를 개선해보면 
        # 그래서 sustain_end 를 release_start지점을 먼저 찾고 -5 한 프레임으로 지정

    # 아래의 main(a, b)
    # print(f"sustain_start: {sustain_start}") #346
    # print(f"sustain_end: {sustain_end}") #370 이라는 수가 나와서 우선 지금은 아래에서 sustain_sample_end 는 370이 나옴

    if sustain_end > sustain_start and (sustain_end - sustain_start) >= 10:
        sustain_sample_end = min(sustain_start + 50, sustain_end)

        # 충분한 sustain 구간
        # 최대 50 프레임만 샘플링
        # ✅ Sustain 구간의 하위 10% 평균
        sustain_region = rms[sustain_start:sustain_sample_end]
        
        # 하위 10% 값들
        bottom_10_percent = np.percentile(sustain_region, 10)
        
        # 또는 하위 10% 평균
        sorted_region = np.sort(sustain_region)
        n_samples = max(1, int(len(sorted_region) * 0.1))
        sustain_val = np.mean(sorted_region[:n_samples])
        
        sustain_present = True

    else:
        #sustain 구간이 부족하다면? (pluck 타입)
        sustain_val = 0
        sustain_present = False 


    # if sustain_end > sustain_start:
        # sustain_val = np.median(rms[sustain_start:sustain_end]) #중간값 찾아서 sustain_val 로 지정
    # else:
        #너무 짧은 소리의 경우, 인덱스 모자람
        # sustain_val = rms[-1] if len(rms) > 0 else rms_peak_val * 0.1


    # 6. Decay 시간 계산
    decay_ms = (times[decay_end] - times[rms_peak_idx]) * 1000 # ms변환 하기 위해 *1000함

    # 7. Sustain 레벨 (dB)
    # peak 대비 상대적 레벨
    if sustain_present and sustain_val > rms_peak_val * 0.01: #peak 의 1%이상
        sustain_db = 20 * np.log10(sustain_val / (rms_peak_val + 1e-8)+ 1e-8)
    else:
        sustain_db = -np.inf #또는 None 으로 처리

    # print(f"sustain_db : {sustain_db}")


    # 8. Release 시작점 찾기
    # 뒤에서부터 threshold 이상인 마지막 지점 
    # threshold = rms_peak_val * 0.01 #Peak 1%
    # release_start = len(rms) - 1

    # for i in range(len(rms)-1, decay_end, -1): #이건 아마도 뒤에서 부터 판단하는걸까?
    #     if rms[i] > threshold:
    #         release_start = i
    #         break

    # 9. Release 시간 계산
    release_ms = (times[-1] - times[release_start])*1000 
        # 시간의 맨 뒤의 값 (전체 시간) - release start 빼고 ms 로 표현하기 위해서 1000 곱합.
        # 근데 전체시간의 뒤의 갑셍서 빼면 안되고, 볼륨이 0이 된 시점까지 계산해야 하는거 아닌가?
    
    if release_ms < 10:
        release_ms = 0

    #Release 확인 
    # print(f"\n===Release확인===")
    # print(f"len(rms) = {len(rms)}")
    # print(f"len(timees) = {len(times)}")
    # print(f"times[-1] = {times[-1]:.6f}초")
    # print(f"release_start = {release_start}")
    # print(f"times[release_start] = {times[release_start]:.6f}초")
    # print(f"차이 = {times[-1] - times[release_start]:.6f}초")
    # print(f"release_ms = {(times[-1] - times[release_start]) * 1000:.1f}ms")

        # times[-1] = 4.000000
        # times[release_start] = 4.000000
        # 이 나옴 그럼 빼면 0 인데... 거기에 1000을 곱하면 0 이 나옴
        # 그렇게 해서 release 가 0ms 가 나오는데, 


    return {
        'attacks_ms' : attack_ms,
        'peak_attack_ms' : peak_attack_ms,
        'decay_ms' : decay_ms,
        'sustain_db' : sustain_db,
        'sustain_present' : sustain_present,
        # 'sustain_frames' : available_frames,
        'release_ms' : release_ms,
        'peak_idx' : rms_peak_idx,
        'decay_end' : decay_end,
        'sustain_start' : sustain_start,
        'release_start' : release_start

    }


# ══════════════════════════════════════════════════════════════
# 함수: 시각화
# ══════════════════════════════════════════════════════════════

def plot_adsr_analysis(y, sr, adsr, hpss, filename, save=False):

    fig, axes = plt.subplots(2, 1, figsize=(8,6))

    #1. adsr 표기
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)

    axes[0].plot(times, rms, color='#333333', linewidth=1)

    #peak 표시
    axes[0].axvline(times[adsr['peak_idx']], color='r',
                    linestyle='--', alpha=0.7, label='Peak(Attack end)')
    
    #Decay 끝 표시
    axes[0].axvline(times[adsr['decay_end']], color='g',
                    linestyle='--', alpha=0.7, label='Decay end')
    
    #Sustain 시작 표시
    if adsr['sustain_start']<len(times):
        axes[0].axvline(times[adsr['sustain_start']], color='b',
                        linestyle='--', alpha=0.7, label='Sustain start')
        
    #Release 시작 표시
    if adsr['release_start']<len(times):
        axes[0].axvline(times[adsr['release_start']], color='purple',
                        linestyle='--', alpha=0.7, label='Release start')
    
    

    axes[0].set_title(f"ADSR Envelope - {filename}")
    axes[0].set_ylabel("RMS energy")
    axes[0].set_xlabel("Time(s)")
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    #2. 원본 파형
    # import librosa.display
    librosa.display.waveshow(y, sr=sr, ax=axes[1], color='#4A90D9')
    axes[1].set_title("Original Waveform")
    axes[1].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()



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
    for path in audio_files[:-1]: #첫번째만 시험해보려 할때는 audio_files[:1] 붙이기
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
                    print(" ℹ️ Stere이지만 L/R 동일 (Left 사용)")
                    y = left
                
                else:
                    # L, R 거의 다름 -> 에너지 합산
                    print(" ℹ️ Stereo 다름 (에너지 합산)")
                    y = np.sqrt(left**2 + right**2) # return 대신 y에 이거 들여보냄


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

            #그래프 그리기 추가 ✅
            hpss = {'per_ratio':0, 'type': 'test'} #임시. 우선 HPSS 안 구함
            plot_adsr_analysis(y, sr, adsr, hpss, name_only, save=False) 

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


"""노트 홀드시간은 알수없다. (sustain = 0 이라면)
=> 오디오만으로는 알 수 없기 때문에 
    sustain 레벨로 판단하여 충분히 크다면 release 판단이 가능하고
    sustain 이 0이라면 추정만 가능 

decay end = min (decay_end + 5, len(rms)) => 둘 중 작은것을 골라야 함 (짧은 소리를 대비하기 위해서)


"""

"""ADSR 만든 사운드가 다르게 나오고 있는 걸 발견했다...
=> 근데 이거 logic 내에서 mono로 변환하니까 소리가 마지막 이 좀 더 커진다던지, 하는 그런 오류가 생김.
 so, stereo 그대로 받아와서 왼쪽만 쓰거나, 오른쪽만 써야함
    + 근데 알고리즘들이 기본적으로 1차원 배열만 처리하기 때문에 ! mono 로 처리를 해줘야함 

    => 따라서 mono로 각각 left 만 처리하거나, right 만 처리해서 합칠 수 도 있고
    => 지금은, 왼쪽만 사용하도록 처리를 하는 방법을 택하자 (right 정보는 버림)

    but, 피아노 같은 악기의 경우, 왼쪽은 low, 오른쪽은 high 이렇게 음역이 나뉘어져 있기때문에 문제가 생길 수 있음! 
"""