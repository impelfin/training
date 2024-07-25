import torch

print("### CUDA 지원 여부 체크 ###")
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
# print(f"CUDA 지원 환경 여부: {torch.cuda.is_built()}")
