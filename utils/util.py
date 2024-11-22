import os
import os.path as osp
import shutil

voc_path = "/home/zyt/Data/VOCdevkit"
voc_ann_path = osp.join(voc_path,"VOC2012/Annotations")
dataset_image_path = osp.join(voc_path, "VOC2012/JPEGImages")
voc_data_path = "/home/zyt/Data/VOCdevkit/VocData"
train_ann_path = osp.join(voc_data_path,"train_ann")
train_image_path = osp.join(voc_data_path,"train_image")
val_ann_path = osp.join(voc_data_path,"val_ann")
val_image_path = osp.join(voc_data_path,"val_image")

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

### copy data
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

if __name__ == '__main__':
    xml_files = get_annpathes(voc_ann_path)
    image_files = get_images(dataset_image_path)
    print(f"images: {len(image_files)},xml_files: {len(xml_files)}")
    limit = 10000
    # split_files(dataset_image_path,train_image_path,val_image_path,limit)
    split_files(voc_ann_path,train_ann_path,val_ann_path,limit)