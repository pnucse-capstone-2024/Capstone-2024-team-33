import requests
import time

POST_URL = "http://192.168.0.118:8000/audio_inference"

def func(file_path, hospital_id=1, mike_id=1):
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f, 'audio/wav')}
        data = {
            'hospital_id': hospital_id,
            'mike_id': mike_id
        }
        
        # 여러 파일을 한 번에 전송
        response = requests.post(POST_URL, files=files, data=data)

    if response.status_code == 200:
        print(f"Response: {response.json()}")
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")


#func('segment_1.wav', 1752, 1)
#time.sleep(5)
#func('segment_2.wav', 1752, 2)
#time.sleep(5)
func('10.낙상_318098_label.wav', 1752, 1)
#time.sleep(5)
# func('10.낙상_317473_label.wav', 1752, 1)
# time.sleep(5)

#func('segment_4.wav', 1752, 4)
#time.sleep(5)
#func('segment_5.wav', 1752, 5)
#time.sleep(5)