from ultralytics import YOLO
import torch

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화
model = YOLO('yolov8n.pt').to(device)

# 학습 설정
data_path = 'snack.yaml' 
epochs = 1 
batch_size = 64  
img_size = 640  

# 모델 학습
model.train(
    data=data_path, 
    epochs=epochs, 
    batch=batch_size, 
    project=".", 
    name="result",  
    exist_ok=True,  
    device=device,
    # cache=True, 
    imgsz=img_size,  
    amp=False,  
    # pretrained=True 
)

print("학습 완료")
