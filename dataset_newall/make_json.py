import os
import json
from glob import glob
from PIL import Image


def create_json(dataset_path, output_file, categories):
    datasets = ['train', 'val', 'test']

    # JSON 구조
    json_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)]
    }

    annotation_id = 0

    for dataset in datasets:
        images_path = os.path.join(dataset_path, dataset, 'images')
        labels_path = os.path.join(dataset_path, dataset, 'labels')

        # 모든 이미지 파일을 찾음
        image_files = glob(os.path.join(images_path, '*.jpg')) + glob(os.path.join(images_path, '*.png'))

        for image_file in image_files:
            # 이미지 파일 이름과 확장자를 분리
            image_name = os.path.basename(image_file)
            label_file = os.path.join(labels_path, os.path.splitext(image_name)[0] + '.txt')

            # 이미지 크기를 읽음
            with Image.open(image_file) as img:
                width, height = img.size

            # 이미지 정보를 JSON 데이터에 추가
            image_info = {
                "file_name": os.path.join(dataset, 'images', image_name),
                "height": height,
                "width": width,
                "id": len(json_data["images"]) + 1
            }
            json_data["images"].append(image_info)

            # 라벨 파일을 읽음
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    category_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:]]

                    # 바운딩 박스를 절대 좌표로 변환
                    x_center, y_center, bbox_width, bbox_height = bbox
                    x_min = (x_center - bbox_width / 2) * width
                    y_min = (y_center - bbox_height / 2) * height
                    bbox_width *= width
                    bbox_height *= height

                    # 어노테이션 정보를 JSON 데이터에 추가
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_info["id"],
                        "category_id": category_id,
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    }
                    json_data["annotations"].append(annotation_info)
                    annotation_id += 1

    # JSON 파일로 저장
    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)


# 데이터셋 경로 및 카테고리 이름
dataset_path = "/Users/lune/Documents/CoreML/dataset_newall"
output_file = "dataset.json"
categories = [
    "0_pokey",
    "1_kkokkalkon",
    "2_Squidzip",
    "3_hotshrimpkkang",
    "4_conecho",
    "5_bananakick",
    "6_kochujang",
    "7_cocacola",
    "8_conetea",
    "9_vita500",
    "10_Fantagrape",
]

# JSON 파일 생성
create_json(dataset_path, output_file, categories)
