import os
import shutil

# 상위 디렉토리 설정
current_directory = '/work/dev_sauron/hsu/opencv_test/mart_data/newdataset/train/labels'

# 폴더 목록
folders = [str(i) for i in range(9)]

for folder in folders:
    folder_path = os.path.join(current_directory, folder)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                shutil.move(file_path, current_directory)
    else:
        print(f"Folder {folder} does not exist")

print("Files moved successfully")
