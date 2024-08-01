from ultralytics import YOLO
import torch

# model = YOLO('yolov8n.pt')
model = YOLO('besta.pt')

device = torch.device("mps")

model.predict(
   source='./input/snac1.png',
   conf=0.25,
   save=True,
   project=".",
   name="result",
   exist_ok=True,
   device=device
)

