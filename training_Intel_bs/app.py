from ultralytics import YOLO
import torch
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 모델 초기화
    model = YOLO('yolov8n.yaml')

    # GPU 사용 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 학습 설정
    data_path = 'dataset.yaml'  # data.yaml 파일 경로
    epochs = 30  # 학습할 에폭 수
    batch_size = 16  # 배치 크기

    # 콜백 함수 정의
    def my_callback(epoch, logs):
        if epoch % 30 == 0:
            model_name = f"model_epoch_{epoch}.pt"
            model.save(model_name)
            print(f"Saved model at {model_name}")

    # 콜백 추가
    model.add_callback('on_epoch_end', my_callback)

    # 모델 학습
    model.train(
        data=data_path, 
        epochs=epochs, 
        batch=batch_size, 
        project=".", 
        name="result", 
        exist_ok=True, 
        device=device,
        imgsz=640
    )
    print("학습 완료")
