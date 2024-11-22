import os
import os.path as osp
import shutil
from os.path import split
import sys
import os.path as osp
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

voc_path = "/home/zyt/Data/VOCdevkit"
voc_ann_path = osp.join(voc_path,"VOC2012/Annotations")
dataset_image_path = osp.join(voc_path, "VOC2012/JPEGImages")
voc_data_path = "/home/zyt/Data/VOCdevkit/VocData"
data_path = osp.join(project_path,"data")
train_file = osp.join(data_path,"train.txt")
val_file = osp.join(data_path,"val.tx")

### get_vocxml_pathes
def get_annpathes(directory):
    xml_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_files.append(osp.join(directory,filename))
    sorted_xml_files = sorted(xml_files)
    return sorted_xml_files

### get_images
def get_images(directory):
    image_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            image_files.append(osp.join(directory,filename))
    sorted_image_files = sorted(image_files)
    return sorted_image_files

### get_files
def get_files(directory):
    files = []
    for filename in os.listdir(directory):
        files.append(osp.join(directory,filename))
    sorted_files = sorted(files)
    return sorted_files

### copy image data
def split_files(src_folder,dest_folder1,dest_folder2,limit):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(dest_folder1, exist_ok=True)
    os.makedirs(dest_folder2, exist_ok=True)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    # 先按照文件名排序（可选，依据需求）
    files.sort()

    # 将前10000个文件复制到dest_folder1
    for i, file_name in enumerate(files[:limit]):
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder1, file_name)
        shutil.copy(src_file, dest_file)

    # 将其余的文件复制到dest_folder2
    for file_name in files[limit:]:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder2, file_name)
        shutil.copy(src_file, dest_file)

### write paths to files
def split_ann_files(src_folder,file1_path, file2_path, limit):
    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # 按照文件名排序（如果需要，可以根据需求调整排序规则）
    files.sort()

    # 文件路径列表
    file_paths1 = []
    file_paths2 = []

    # 将前10000个文件路径存入file_paths1
    for i, file_name in enumerate(files[:limit]):
        file_path = os.path.join(src_folder, file_name)
        file_paths1.append(file_path)

    # 将其余的文件路径存入file_paths2
    for file_name in files[limit:]:
        file_path = os.path.join(src_folder, file_name)
        file_paths2.append(file_path)

    # 将文件路径写入file1.txt
    with open(file1_path, 'w') as file1:
        for path in file_paths1:
            file1.write(path + '\n')  # 每个路径占一行

    # 将文件路径写入file2.txt
    with open(file2_path, 'w') as file2:
        for path in file_paths2:
            file2.write(path + '\n')  # 每个路径占一行

### split train and val
def split_data(src_folder,file1_path, file2_path, limit):
    # 获取源文件夹中的所有文件
    files = [osp.splitext(f)[0] for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # 按照文件名排序（如果需要，可以根据需求调整排序规则）
    files.sort()

    # 文件路径列表
    file_paths1 = files[:limit]
    file_paths2 = files[limit:]
    print(f"file1: {len(file_paths1)},file2: {len(file_paths2)}")
    # 将文件路径写入file1.txt
    with open(file1_path, 'w') as file1:
        for path in file_paths1:
            file1.write(path + '\n')  # 每个路径占一行

    # 将文件路径写入file2.txt
    with open(file2_path, 'w') as file2:
        for path in file_paths2:
            file2.write(path + '\n')  # 每个路径占一行

if __name__ == '__main__':
    xml_files = get_annpathes(voc_ann_path)
    image_files = get_images(dataset_image_path)
    print(f"images: {len(image_files)},xml_files: {len(xml_files)}")
    limit = 10000
    # split_files(dataset_image_path,train_image_path,val_image_path,limit)
    split_data(voc_ann_path, train_file, val_file, limit)
