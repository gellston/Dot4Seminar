import torch


from model.zerodcepp import ZeroDCEPP



# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")




# Hyper Parameter
learning_rate = 0.003
batch_size = 5
scale_factor = 4
epochs = 100
num_features = 32



model = ZeroDCEPP(scale_factor=scale_factor, num_features=num_features)
model = model.to(device)
