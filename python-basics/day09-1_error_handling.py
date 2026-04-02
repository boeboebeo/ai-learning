# Day 09 - Error Handling
import json

def get_bpm_category(bpm):
    try:
        bpm = float(bpm)
        if bpm <= 0:
            raise ValueError("BPM must be greater than 0")
        if bpm > 300:
            raise ValueError("BPM seems too high")
        if bpm < 80:
            return "Slow"
        elif bpm < 120:
            return "Mid-tempo"
        else:
            return "Fast"
    except ValueError as e:
        return f"Error: {e}"
    except TypeError:
        return "Error: BPM must be a number"
    
    #try : 에러가 날 수 있는 코드
    #except : 에러가 나면 여기로 와라! -> 에러가 떠도 프로그램이 바로 죽지않고 계속 실행될 수 있게 함

def load_playlist(filename):
    try:    #에러날 수 도 있으니 시도해봐 -> 에러나면 except 로 가줘
        with open(filename, "r") as f:  #파일 열어서 f 로 부를게 
            return json.load(f)     #열린 파일을 파이썬으로 변환해서 리턴
    except FileNotFoundError:
        print(f"❌ File '{filename}' not found")
        return [] #빈 리스트 라도 출력!
    except json.JSONDecodeError:
        print(f"❌ '{filename}' is not valid JSON")
        return []

print("=== BPM Tests ===")
test_values = [120, 67, -10, 500, "abc", None]
for val in test_values:
    print(f"Input: {val} → {get_bpm_category(val)}")

print("\n=== File Loading ===")
playlist = load_playlist("playlist.json")
if playlist:
    print(f"✅ Loaded {len(playlist)} songs")

print("\n", get_bpm_category(-2))
print(get_bpm_category(305))


# 1) 이렇게 하면 좀 지저분하게 출력되어서 
print(load_playlist("playlist.json")) #여기는 try 해서 에러안난다면 print 가 없기떄문에 print 해줘야 함 (에러난 곳에는 print 존재)

# 2) 깔끔하게 for 문으로 정리한 버전 (나중에 또 쓸거면 그냥 변수에 담아서 쓴느게 편함)
result = load_playlist("playlist.json") 
print("\n")
for song in result:
    
    print(f"- {song['title']} by {song['artist']} ({song['bpm']})")
