import torch
import torch.onnx
import numpy as np
import cv2
import os


from model.zerodcepp import ZeroDCEPP
from loss.zerodce_loss import ZeroDCETotalLoss
from dataset.lle_dataset import get_lle_loader
from model.onnx_model import OnnxModel



# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")




# Hyper Parameter
epochs = 50
learning_rate = 0.0001
weight_decay = 0.0001
batch_size = 8
scale_factor = 4
num_features = 32
image_width = 1024
image_height = 1024
image_channel = 3

dataset_path = "C://github//dataset//lol_dataset//our485//all"
weight_path = "C://github//Dot4Seminar//working//python//results//weights.pth"
onnx_model_path = "C://github//Dot4Seminar//working//python//results//model.onnx"


dummy_input = torch.randn(size=(1, image_channel, image_height, image_width)).to(device)

model = ZeroDCEPP(scale_factor=scale_factor, num_features=num_features)
model = model.to(device)
if os.path.exists(weight_path):
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)

model.eval()
