import coremltools as ct
import numpy as np
from PIL import Image, ImageDraw
import os

# 모델 로드
model = ct.models.MLModel('your_model.mlmodel')

# 입력 및 출력 디렉토리 설정
input_folder = 'input'
output_folder = 'result'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 처리 함수
def process_image(image_path, model):
    image = Image.open(image_path)
    image = image.convert("RGB")
    input_width, input_height = image.size

    # 모델 입력 준비
    input_data = {
        'image': image
    }
    
    # 예측 실행
    results = model.predict(input_data)

    # 바운딩 박스 추출
    boxes = results['coordinates']
    scores = results['confidence']
    labels = results['classLabel']

    # 바운딩 박스 그리기
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # 신뢰도 기준 (0.5 이상만 표시)
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * input_width)
            y_min = int(y_min * input_height)
            x_max = int(x_max * input_width)
            y_max = int(y_max * input_height)
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
            draw.text((x_min, y_min), f'{label} {score:.2f}', fill='red')

    return image

# 디렉토리 내 이미지 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        result_image = process_image(input_path, model)
        result_image.save(output_path)

print("모든 이미지를 처리하고 결과를 저장했습니다.")
