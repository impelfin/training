import coremltools as ct
import numpy as np
from PIL import Image, ImageDraw
import os

# 모델 로드
model = ct.models.MLModel('sauron_snack.mlmodel')

# 모델이 허용하는 입력 크기
input_width, input_height = 416, 416  # 여기에 모델이 허용하는 크기를 설정합니다

# 입력 및 출력 디렉토리 설정
input_folder = 'input'
output_folder = 'result'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 클래스 이름 리스트 (모델에 맞게 수정 필요)
class_labels = [
    'orion_Pocachip_Original_66G',
    'orion_Gosomi_80G',
    'orion_Fresh_Berry_336G',
    'orion_Diget_tongmil_28_194G',
    'orion_Diget_Choco_312G',
    'orion_chokchokhan_Chocochip_240G',
    'orion_Chocolate_Chip_Cookies_256G',
    'nongshim_Ojingeojip_83G',
    'nongshim_ChipPotato_Original_125G',
    'nongshim_Banana_Kick_75G',
    'nongshim_Alsaeuchip_68G',
    'lotte_kkokkalkon_gosohanmas_72G',
    'haetae_Oyeseu_360G',
    'haetae_Osajjeu_60G',
    'haetae_Masdongsan_90G',
    'haetae_HoneyButterChip_38G',
    'haetae_Guun_Gamja_162G',
    'crown_Potto_Cheese_Tart_322G',
    'crown_Concho_66G',
    'crown_ChocoHaim_142G',
    'crown_BigPie_Strawberry_324G'
]

# 이미지 처리 함수
def process_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    image_resized = image.resize((input_width, input_height))  # 이미지 크기 조정

    # 모델 입력 준비
    input_data = {
        'imagePath': image_resized,  # 이미지 객체를 'imagePath'로 전달
        'confidenceThreshold': 0.5,
        'iouThreshold': 0.5
    }
    
    # 예측 실행
    try:
        results = model.predict(input_data)
        print(results)  # 예측 결과 출력
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None

    # 바운딩 박스 추출
    boxes = results['coordinates']
    confidences = results['confidence']

    # 바운딩 박스를 원래 이미지 크기에 맞게 조정
    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        # 가장 높은 확률의 클래스 선택
        max_confidence_index = np.argmax(confidences[i])
        
        # 클래스 인덱스가 올바른지 확인
        if max_confidence_index < len(class_labels):
            score = confidences[i][max_confidence_index]
            if score > 0.5:  # 신뢰도 기준 (0.5 이상만 표시)
                # 바운딩 박스 좌표 변환
                x_min, y_min, x_max, y_max = box
                x_min = x_min * original_width
                y_min = y_min * original_height
                x_max = x_max * original_width
                y_max = y_max * original_height

                # 좌표 값 교정 (x_min, y_min < x_max, y_max)
                if x_min > x_max:
                    x_min, x_max = x_max, x_min
                if y_min > y_max:
                    y_min, y_max = y_max, y_min

                label = class_labels[max_confidence_index]
                draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=3)
                draw.text((x_min, y_min), f'{label} {score:.2f}', fill='red')
        else:
            print(f"Warning: max_confidence_index {max_confidence_index} is out of range for class_labels")

    return image

# 디렉토리 내 이미지 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        result_image = process_image(input_path, model)
        if result_image:
            result_image.save(output_path)

print("모든 이미지를 처리하고 결과를 저장했습니다.")
