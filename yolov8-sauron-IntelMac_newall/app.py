import os
from ultralytics import YOLO

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Load the model
model = YOLO('besti.pt')

results = model.predict(
    source='./input/in2.png',
    conf=0.25,
    save=True,
    project=".",
    name="result",
    exist_ok=True
)
