import os
import json
import shutil
import random

def split_dataset(train_folder, test_folder, json_file, test_split=0.3):
    # JSON 파일 열기
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 이미지 파일 목록
    image_files = [item['image'] for item in data]  # 여기서 'imagefilename' 대신 'image' 사용

    # 섞기
    random.shuffle(image_files)

    # 데이터셋 분할
    split_index = int(len(image_files) * test_split)
    test_files = image_files[:split_index]
    train_files = image_files[split_index:]

    # 폴더 생성
    os.makedirs(test_folder, exist_ok=True)

    # 파일 이동 및 어노테이션 나누기
    test_data = []
    train_data = []

    for item in data:
        image_file = item['image']  # 여기서도 'imagefilename' 대신 'image' 사용
        if image_file in test_files:
            shutil.move(os.path.join(train_folder, image_file), os.path.join(test_folder, image_file))
            test_data.append(item)
        else:
            train_data.append(item)

    # JSON 파일 저장
    with open(os.path.join(train_folder, 'annotations_train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(test_folder, 'annotations_test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"Dataset split completed. {len(train_data)} images in train set, {len(test_data)} images in test set.")

# 경로 설정
train_folder = "train"  # 학습 이미지 폴더 경로
test_folder = "test"  # 테스트 이미지 폴더 경로
json_file = os.path.join(train_folder, "annotations.json")  # 입력 JSON 파일 경로

# 데이터셋 분할 실행
split_dataset(train_folder, test_folder, json_file)
