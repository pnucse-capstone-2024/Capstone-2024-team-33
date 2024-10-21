import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.thermal.conv3d_l import Conv3dL
from fastapi import FastAPI, UploadFile, File, Form
import torchvision.transforms as transforms
import io
from PIL import Image
import numpy as np
from fastapi.responses import JSONResponse
from typing import List
from torch2trt import torch2trt
import librosa
from models.audio.custom_efficientnetb0 import CustomEfficientNetB0
from pydub import AudioSegment
import websockets
import json
import asyncio
import base64
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("[Log] Thermal model loading...")
thermal_model = Conv3dL()
thermal_model.load_state_dict(torch.load('/home/insung/ai_server/weights/thermal/conv3d_l.pt'))
thermal_model.eval()
thermal_model.to(device)
x = torch.ones((1, 1, 50, 64, 64)).to(device)
thermal_model_trt = torch2trt(thermal_model, [x])
print("[Log] Thermal model is loaded.")

print("[Log] Audio model loading...")
audio_model = CustomEfficientNetB0()
audio_model.load_state_dict(torch.load('/home/insung/ai_server/weights/audio/efficientnetb0.pt'))
audio_model.eval()
audio_model.to(device)
x = torch.ones((1, 1, 224, 224)).to(device)
audio_model_trt = torch2trt(audio_model, [x])
print("[Log] Audio model is loaded.")

print("[Log] FastAPI loading...")
app = FastAPI()
print("[Log] FastAPI is loaded.")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),        
])


websocket_image_connection = None

@app.on_event("startup")
async def startup_event():
    """Establish WebSocket connection when FastAPI starts."""
    global websocket_image_connection
    image_uri = "wss://api.teameffective.link/image-stream"
    
    try:
        websocket_image_connection = await websockets.connect(image_uri, ping_interval=1000, ping_timeout=1000)
        print("WebSocket image connection established.")
    except Exception as e:
        print(f"Failed to establish WebSocket image connection: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close WebSocket connection when FastAPI shuts down."""
    global websocket_image_connection
    if websocket_image_connection:
        await websocket_image_connection.close()

async def reconnect_websocket():
    global websocket_image_connection
    image_uri = "wss://api.teameffective.link/image-stream"
    
    while True:
        try:
            websocket_image_connection = await websockets.connect(image_uri, ping_interval=1000, ping_timeout=1000)
            print("WebSocket reconnected.")
            break  # 성공적으로 재연결 시 while 루프 종료
        except Exception as e:
            print(f"Failed to reconnect WebSocket: {e}")
            await asyncio.sleep(5)  # 재연결 시도 전 잠시 대기

async def send_and_receive(websocket, message):
    try:
        await websocket.send(message)
    except websockets.ConnectionClosedError as e:
        print(f"WebSocket connection closed with error: {e}")
        await reconnect_websocket()
    except Exception as e:
        print(f"Error during WebSocket communication: {e}")

@app.post("/audio_inference")
async def audio_inference(
    file: UploadFile = File(...),
    hospital_id: str = Form(...),
    mike_id: str = Form(...)  # 문자열 데이터 처리
):
    model_duration = 5.0

    print(f"Received file: {file.filename}")

        # Read the audio file
    audio_bytes = await file.read()  # Read the content of the file
    audio_io = io.BytesIO(audio_bytes)  # Convert to a BytesIO stream
        
        # Use librosa to load the audio from the byte stream
        # `sr=None` ensures that the original sample rate is preserved
    try:
        audio = AudioSegment.from_file(audio_io)
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
    except Exception as e:
        return JSONResponse({"error": f"Error processing audio file"})

    
    y, sr = librosa.load(wav_io, sr=44100)
    duration = librosa.get_duration(y=y, sr=sr)
            
    if duration > model_duration:
        num_samples_to_keep = int(model_duration * sr)
        y = y[:num_samples_to_keep]
        duration = model_duration
        
    S = librosa.feature.mfcc(
        y=y, 
        sr=sr,
        n_mfcc=40
    )

    S = cv2.resize(S, (224, 224))


    S = torch.FloatTensor(S)
    S = S.unsqueeze(0)
    S = S.unsqueeze(0)


    with torch.no_grad():
        S = S.to(device)
        logit = audio_model_trt(S)
        pred = logit.argmax(1).detach().cpu().numpy()
    
    print(f"pred = {pred}")

    if pred == 0: # 사고가 발생했을때
        try:
            data = {
                "type": "audioAccident",
                "hospitalId": hospital_id,
                "mikeId": mike_id
            }
            json_data = json.dumps(data)
            await send_and_receive(websocket_image_connection, json_data)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")
    
    return JSONResponse({"pred": pred.tolist()})
    

@app.post("/thermal_inference") # 5초마다 호출되는 함수임.
async def thermal_inference(
    files: List[UploadFile] = File(...),
    hospital_id: str = Form(...),
    camera_id: str = Form(...)
):
    frames = []
    for file in files:
        print(f"Received file: {file.filename}")
        file_content = await file.read()

        # 이미지 처리
        try:
            frame = Image.open(io.BytesIO(file_content))
            frames.append(transform(frame))

            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")  # 이미지 포맷에 맞게 저장
            base64_encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

            data = {
                "type": "image",
                "encodedImage": base64_encoded_image,
                "hospitalId": hospital_id,
                "cameraId": camera_id
            }
            json_data = json.dumps(data)
            await send_and_receive(websocket_image_connection, json_data)
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")

    frames = torch.stack(frames)  
    frames = frames.unsqueeze(0)  
    frames = frames.permute(0, 2, 1, 3, 4)  

    with torch.no_grad():
        frames = frames.to(device)
        logit = thermal_model_trt(frames)
        pred = logit.argmax(1).detach().cpu().numpy()
    
    print(f"pred = {pred}")

    if pred == 0: # 사고가 발생했을때
        try:
            data = {
                "type": "thermalAccident",
                "hospitalId": hospital_id,
                "cameraId": camera_id
            }
            json_data = json.dumps(data)
            await send_and_receive(websocket_image_connection, json_data)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")
    
    return JSONResponse({"pred": pred.tolist()})

@app.get("/websocket-test")
async def my_post(
    hospital_id: str = Form(...),
    camera_id: str = Form(...)
):
    frame_folder = "fall"
    frame_count = 50
      
    frames = []
    for idx in range(frame_count):
        filename = f"{frame_folder}/{idx}.png"
        with open(filename, 'rb') as file:
            file_content = file.read()
            frame = Image.open(io.BytesIO(file_content))
            frames.append(transform(frame))
            try:
                buffered = io.BytesIO()
                frame.save(buffered, format="PNG")  # 이미지 포맷에 맞게 저장
                base64_encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                data = {
                    "type": "image",
                    "encodedImage": base64_encoded_image,
                    "hospitalId": hospital_id,
                    "cameraId": camera_id
                }
                json_data = json.dumps(data)
                await send_and_receive(websocket_image_connection, json_data)
                await asyncio.sleep(0.3)
            except websockets.ConnectionClosedError as e:
                print(f"WebSocket connection closed with error: {e}")
            except Exception as e:
                print(f"Error during WebSocket communication: {e}")

    frames = torch.stack(frames)  
    frames = frames.unsqueeze(0)  
    frames = frames.permute(0, 2, 1, 3, 4)  

    with torch.no_grad():
        frames = frames.to(device)
        logit = thermal_model_trt(frames)
        pred = logit.argmax(1).detach().cpu().numpy()
    
    print(f"pred = {pred}")

    if pred == 0: # 사고가 발생했을때
        try:
            data = {
                "type": "thermalAccident",
                "hospitalId": hospital_id,
                "cameraId": camera_id
            }
            json_data = json.dumps(data)
            await send_and_receive(websocket_image_connection, json_data)
        except Exception as e:
            print(f"Error during WebSocket communication: {e}")
    
    return JSONResponse({"pred": pred.tolist()})


@app.get("/")
async def root():
    return {"message": "Hello World"}