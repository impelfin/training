import os

img_folder = 'images/'
label_folder = 'labels/'
new_class_num = 5

def rename_and_update_files(new_class_num):
    # Rename files in img folder
    img_files = sorted(os.listdir(img_folder))
    for index, file_name in enumerate(img_files):
        if not file_name.endswith(('jpg', 'png', 'jpeg')):  # Assuming image files have these extensions
            continue
        new_name = f"{new_class_num}_{index}{os.path.splitext(file_name)[1]}"
        old_file_path = os.path.join(img_folder, file_name)
        new_file_path = os.path.join(img_folder, new_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed image file: {old_file_path} -> {new_file_path}")

    # Rename files and update content in label folder
    label_files = sorted(os.listdir(label_folder))
    for index, file_name in enumerate(label_files):
        if not file_name.endswith('.txt'):
            continue
        new_name = f"{new_class_num}_{index}.txt"
        old_file_path = os.path.join(label_folder, file_name)
        new_file_path = os.path.join(label_folder, new_name)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed label file: {old_file_path} -> {new_file_path}")

        # Update content of the label file
        with open(new_file_path, 'r') as file:
            lines = file.readlines()

        with open(new_file_path, 'w') as file:
            for line in lines:
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    parts[0] = str(new_class_num)
                    file.write(' '.join(parts))
                else:
                    file.write(line)
        print(f"Updated content in file: {new_file_path}")

# Example usage
rename_and_update_files(new_class_num)  
