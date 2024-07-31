from ultralytics import YOLO
import torch
import time

if __name__ == '__main__':
    # 모델 초기화
    model = YOLO('yolov8n.pt')
    # model = YOLO('/Users/lune/Documents/GitHub/training/training_M1_new/result/weights/last.pt')

    # torch 데이터 유형, 백엔드 설정
    dtype = torch.float
    device = torch.device("mps")

    # 학습 설정
    data_path = 'seesaw.yaml'
    epochs = 10
    batch_size = 32

    # 모델 학습
    train_start = time.time()
    model.train(
        data=data_path, 
        epochs=epochs, 
        batch=batch_size, 
        project=".", 
        name="result", 
        exist_ok=True, 
        device=device, 
        optimizer='auto', 
        cache=True, 
        imgsz=640,
        # pretrained=True    
    )
    train_end = time.time()
    
    # 학습 소요 시간 확인
    print()
    print(device.type, '학습 소요시간 : ', train_end - train_start)
