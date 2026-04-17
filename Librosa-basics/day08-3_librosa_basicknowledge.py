# term 

"""(1) len(rms)

rms = librosa.feature.rms(y=y, hop_length=hop)[0] 인데 hop = 512 라면 

rms 는 프레임 단위로 계산됨 

    총 프레임 수 = 오디오 샘플 수 / hop_length
    ex. 오디오 길이 y = 96256 samples (sr = 48000), hop_length = 512

        프레임 수 = 96256 / 512 = 188 프레임
            => len(rms) = 188


            ** y = 원본 오디오 샘플 수 
            ** d = RMS 프레임 간의 시간 간격. = hop / sr = 512 / 48000 = 0.010667
            => rms 는 아래와 같은 시간 간격으로 각각의 프레임으로 출력됨 



"""

d = round(512 / 48000, 6)
for i in range(10):
    times = round(i*d, 8)
    print(times)

    # 0.0
    # 0.010667
    # 0.021334
    # 0.032001
    # 0.042668
    # 0.053335
    # 0.064002
    # 0.074669
    # 0.085336
    # 0.096003


"""(2) rms_fft = np.fft.rfft(rms_normalized) 현재 입력 : 188개

**rms 는 실수 신호의 절반 + 1만 반환: (주파수가 대칭이므로, 양의 주파수만 저장 (Nyquist 정리))
    rfft 출력 크기 : (입력크기 / 2) + 1

    ex. 입력: rms = 188개 .
        출력: 188 / 2 + 1 = 94 + 1 = 95개

        => 따라서
            len(rms_fft) = 95, len(rms_freqs) = 95 
            => 1:1 대응 할 수 있음

"""