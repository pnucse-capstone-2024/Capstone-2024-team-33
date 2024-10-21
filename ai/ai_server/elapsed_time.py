import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models.thermal.conv3d_m import Conv3dM
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
import time





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# thermal_model = Conv3dM()
# thermal_model.load_state_dict(torch.load('/home/insung/ai_server/weights/thermal/conv3d_m.pt'))
thermal_model = Conv3dL()
thermal_model.load_state_dict(torch.load('/home/insung/ai_server/weights/thermal/conv3d_l.pt'))

thermal_model.eval()
thermal_model.to(device)

t = []
for i in range(6):
    start = time.time()
    with torch.no_grad():
        frames = torch.ones((1, 1, 50, 64, 64)).to(device)
        logit = thermal_model(frames)
        pred = logit.argmax(1).detach().cpu().numpy()
        end = time.time()
        if i == 0:
            continue
        t.append(end-start)
print(t)
print(sum(t)/len(t))

x = torch.ones((1, 1, 50, 64, 64)).to(device)
thermal_model_trt = torch2trt(thermal_model, [x])

t = []
for i in range(6):
    start = time.time()
    with torch.no_grad():
        frames = torch.ones((1, 1, 50, 64, 64)).to(device)
        logit = thermal_model_trt(frames)
        pred = logit.argmax(1).detach().cpu().numpy()
        end = time.time()
        if i == 0:
            continue
        t.append(end-start)
print(t)
print(sum(t)/len(t))


# audio_model = CustomEfficientNetB0()
# audio_model.load_state_dict(torch.load('/home/insung/ai_server/weights/audio/efficientnetb0.pt'))
# audio_model.eval()
# audio_model.to(device)

# t = []
# for i in range(6):
#     start = time.time()
#     with torch.no_grad():
#         S = torch.ones((1, 1, 40, 256)).to(device)
#         logit = audio_model(S)
#         pred = logit.argmax(1).detach().cpu().numpy()
#         end = time.time()
#         if i == 0:
#             continue
#         t.append(end-start)
# print(t)
# print(sum(t)/len(t))


# x = torch.ones((1, 1, 40, 256)).to(device)
# audio_model_trt = torch2trt(audio_model, [x])

# t = []
# for i in range(6):
#     start = time.time()
#     with torch.no_grad():
#         S = torch.ones((1, 1, 40, 256)).to(device)
#         logit = audio_model_trt(S)
#         pred = logit.argmax(1).detach().cpu().numpy()
#         end = time.time()
#         if i == 0:
#             continue
#         t.append(end-start)
# print(t)
# print(sum(t)/len(t))