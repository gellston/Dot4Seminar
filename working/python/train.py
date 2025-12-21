import torch
import torch.onnx



from model.zerodcepp import ZeroDCEPP
from loss.zerodce_loss import ZeroDCETotalLoss
from dataset.lle_dataset import get_lle_loader



# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")




# Hyper Parameter
epochs = 100
learning_rate = 0.003
weight_decay = 0.0001
batch_size = 5
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



dataloader = get_lle_loader(dataset_path, batch_size, resize_shape=(image_height, image_width))
total_batches = len(dataloader)


loss = ZeroDCETotalLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

temp_loss = 1000000
for epoch in range(epochs):
    avg_loss = 0

    model.train()
    for i, x_image in enumerate(dataloader):


        gpu_x_image = x_image.to(device)
        enhanced_img, curve_params = model(gpu_x_image)
        current_loss = loss(gpu_x_image, enhanced_img, curve_params)



        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()

        avg_loss += current_loss.item() / total_batches

    if temp_loss > avg_loss:
        temp_loss = avg_loss
        
        model.eval()

        torch.save(model.state_dict(), weight_path)

        torch.onnx.export(
            model,                      # 실행할 모델
            dummy_input,                # 모델 입력 예시
            onnx_model_path,            # 저장 파일명
            export_params=True,         # 모델 파일 안에 학습된 파라미터 저장
            opset_version=11,           # Bilinear 연산을 안정적으로 지원하는 버전
            do_constant_folding=True,   # 상수 폴딩 최적화 (속도 향상)
            input_names=['input'],      # 입력 노드 이름 (C++에서 호출 시 사용)
            output_names=['output'],    # 출력 노드 이름
        )



    print('current avg loss = ', avg_loss)



