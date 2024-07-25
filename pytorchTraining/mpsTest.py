import torch

print("### MPS 지원 여부 체크 ###")
print(f"MPS 사용 가능 여부: {torch.backends.mps.is_available()}")
print(f"MPS 지원 환경 여부: {torch.backends.mps.is_built()}")

print()

print("### MPS 이용 학습 ###")

# 추가 모듈 로드
import torch.nn as nn
import time

# torch 데이터 유형, 백엔드 설정
dtype = torch.float
device = torch.device("mps")

# 3개 변수를 가진 다중회귀 모형 샘플 데이터 설정
x = torch.randint(10, (100, 3),device = device, dtype=dtype)
y = torch.randint(100, 200, (100, 1),device = device, dtype=dtype)

# 선형 모델 구성
ml_model = nn.Linear(3,1) # 3x1 (변수 할당)
ml_model.to(device)

# 옵티마이저
optimizer = torch.optim.SGD(ml_model.parameters(), lr = 1e-3)

# epoch 설정
total_epoch = 3000

# 학습 시작
train_start = time.time()
for epoch in range(total_epoch+1):

    # 예측값
    prediction = ml_model(x)

    # 비용 (Class가 아니므로 nn.functional 사용)
    loss = nn.functional.mse_loss(prediction, y)

    # 역전파
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 중간 기록
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{total_epoch},\t loss : {loss.item()}")

train_end = time.time()

# 학습 소요 시간 확인
print()
print(device.type, '학습 소요시간 ',train_end-train_start)

