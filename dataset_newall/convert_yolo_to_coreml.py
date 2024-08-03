import os
import json
from PIL import Image

def yolo_to_custom_format(images_folder, labels_folder, output_file):
    data = []

    for image_file in os.listdir(images_folder):
        if image_file.endswith(".jpg") or image_file.endswith(".png"):
            image_path = os.path.join(images_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            label_path = os.path.join(labels_folder, label_file)

            if not os.path.exists(label_path):
                print(f"Warning: Label file for {image_file} not found, skipping.")
                continue

            # 이미지 크기 가져오기
            with Image.open(image_path) as img:
                image_width, image_height = img.size

            with open(label_path, "r") as f:
                image_annotations = []
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    annotation = {
                        "label": str(int(class_id)),
                        "coordinates": {
                            "x": x_center * image_width,
                            "y": y_center * image_height,
                            "width": width * image_width,
                            "height": height * image_height
                        }
                    }
                    image_annotations.append(annotation)

                data.append({
                    "imagefilename": image_file,
                    "annotations": image_annotations
                })

    # JSON 파일로 어노테이션 작성
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Annotations successfully written to {output_file}")

# 경로 설정
images_folder = "images"  # 이미지 폴더 경로
labels_folder = "labels"  # 라벨 폴더 경로
output_file = "annotations.json"  # 출력 JSON 파일 경로

# 변환 실행
yolo_to_custom_format(images_folder, labels_folder, output_file)
