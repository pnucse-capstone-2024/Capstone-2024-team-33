import requests
import time

def send_frames_batch(frame_folder="", frame_count=50, hospital_id=1, camera_id=1):
    url = "http://192.168.0.118:8000/thermal_inference"
      
    # 여러 파일을 한 번에 전송할 수 있도록 파일 리스트 생성
    files = []
    for idx in range(frame_count):
        filename = f"{frame_folder}/{idx}.png"
        with open(filename, 'rb') as file:
            # append 대신 딕셔너리에서 key-value로 여러 파일 전송
            files.append(('files', (filename, file.read(), 'image/png')))

    data = {
        'hospital_id': hospital_id,
        'camera_id': camera_id
    }
    
    # 여러 파일을 한 번에 전송
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")

# 여러 프레임을 한 번에 전송
send_frames_batch("./39", 50, 1752, 2)
time.sleep(5)

send_frames_batch("./40", 50, 1752, 3)
time.sleep(5)

send_frames_batch("./41", 50, 1752, 4)
time.sleep(5)

send_frames_batch("./42", 50, 1752, 5)
time.sleep(5)

send_frames_batch("./43", 50, 1752, 6)
time.sleep(5)

# 0: 낙상 1: 정상