import sys
import os.path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

from core.dataset import recover_input

project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from net.centernet import CenterNet
from core.helper import get_model,remove_dir_and_create_dir,get_dataset
from util import *
from tools.train import CosineAnnealingWarmupRestarts
from tools.args import args, dev, class_names
from core.dataset.utils import image_resize, preprocess_input, gaussian_radius, draw_gaussian
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

test_save_path = osp.join(project_path,"test_save_data","image")

def transform_image_bboxes():
    train_dataset, val_dataset = get_dataset(args, class_names)
    print(f"train_dataset: {len(train_dataset)},val_dataset: {len(val_dataset)}")
    img_idx = 100
    input_shape = (args.input_height, args.input_width)

    #### geometric transformation
    image, bboxes = train_dataset.parse_annotation(img_idx)
    print(f"[ RawImage ],img_idx:{img_idx},image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)
    image,bboxes = train_dataset.random_horizontal_flip(image,bboxes)
    print(f"[ HorizonFlip ],image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)
    image,bboxes = train_dataset.random_vertical_flip(image,bboxes)
    print(f"[ VerticalFlip ],image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)
    image,bboxes = train_dataset.random_crop(image,bboxes)
    print(f"[ RandomCrop ],image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)
    image,bboxes = train_dataset.random_translate(image,bboxes)
    print(f"[ RandomTranslate ],image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)
    image,bboxes = image_resize(image,input_shape,bboxes)
    print(f"[ Resize ],image: {image.shape},bboxes: {bboxes.shape}")
    image = preprocess_input(image)
    print(f"unit_image: {image.shape},mean: {np.mean(image,axis=(1,2))},std: {np.std(image,axis=(1,2))}")

def ann_to_label():
    train_dataset, val_dataset = get_dataset(args, class_names)
    print(f"train_dataset: {len(train_dataset)},val_dataset: {len(val_dataset)}")
    img_idx = 100
    input_shape = (args.input_height, args.input_width)
    stride = 4
    output_shape = (input_shape[0] // stride, input_shape[1] // stride)

    # region transform
    image, bboxes = train_dataset.parse_annotation(img_idx)
    image, bboxes = train_dataset.data_augmentation(image, bboxes)
    image, bboxes = image_resize(image, train_dataset.input_shape, bboxes)
    image = preprocess_input(image)
    # endregion

    # region Clip bounding boxes
    clip_bboxes = []
    labels = []
    for bbox in bboxes:
        x1, y1, x2, y2, label = bbox

        if x2 <= x1 or y2 <= y1:
            # Don't use such boxes as this may cause nan loss.
            continue

        x1 = int(np.clip(x1, 0, input_shape[1]))
        y1 = int(np.clip(y1, 0, input_shape[0]))
        x2 = int(np.clip(x2, 0, input_shape[1]))
        y2 = int(np.clip(y2, 0, input_shape[0]))
        # Clipping coordinates between 0 to image dimensions as negative values
        # or values greater than image dimensions may cause nan loss.
        clip_bboxes.append([x1, y1, x2, y2])
        labels.append(label)

    bboxes = np.array(clip_bboxes)
    labels = np.array(labels)
    # endregion
    if len(bboxes) != 0:
        labels = np.array(labels, dtype=np.float32)
        bboxes = np.array(bboxes[:, :4], dtype=np.float32)
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]] / stride, a_min=0, a_max=output_shape[1])
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]] / stride, a_min=0, a_max=output_shape[0])

    batch_hm = np.zeros((output_shape[0], output_shape[1], len(class_names)), dtype=np.float32)
    batch_wh = np.zeros((output_shape[0], output_shape[1], 2), dtype=np.float32)
    batch_offset = np.zeros((output_shape[0], output_shape[1], 2), dtype=np.float32)
    batch_offset_mask = np.zeros((output_shape[0], output_shape[1]), dtype=np.float32)

    for i in range(len(labels)):
        x1, y1, x2, y2 = bboxes[i]
        cls_id = int(labels[i])

        h, w = y2 - y1, x2 - x1
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            # Calculates the feature points of the real box
            ct = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            # Get gaussian heat map
            batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

            # Assign ground truth height and width
            batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h

            # Assign center point offset
            batch_offset[ct_int[1], ct_int[0]] = ct - ct_int

            # Set the corresponding mask to 1
            batch_offset_mask[ct_int[1], ct_int[0]] = 1

def test_train():
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "weights"), is_remove=True)
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "summary"), is_remove=True)

    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args, class_names)
    print(f"train_dataset: {len(train_dataset)},val_dataset: {len(val_dataset)}")
    writer = SummaryWriter(os.path.join(args.logs_dir, "summary"))

    freeze_step = len(train_dataset) // args.freeze_batch_size
    unfreeze_step = len(train_dataset) // args.unfreeze_batch_size
    print(f"freeze_batch_size: {args.freeze_batch_size},freezee_step: {freeze_step}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, args.learn_rate_init)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=args.freeze_epochs * freeze_step + args.unfreeze_epochs * unfreeze_step,
                                              max_lr=args.learn_rate_init,
                                              min_lr=args.learn_rate_end,
                                              warmup_steps=args.warmup_epochs * freeze_step)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.freeze_batch_size,
                              num_workers=args.num_workers, pin_memory=True)

    ### predict
    model.train()

    image_size = len(train_loader)
    print(f"image_size: {image_size}")

def draw_bboxes():
    train_dataset, val_dataset = get_dataset(args, class_names)
    print(f"train_dataset: {len(train_dataset)},val_dataset: {len(val_dataset)}")
    img_idx = 90
    input_shape = (args.input_height, args.input_width)

    #### geometric transformation
    image, bboxes = train_dataset.parse_annotation(img_idx)
    print(f"[ RawImage ],img_idx:{img_idx},image: {image.shape},bboxes: {bboxes.shape}")
    print(bboxes)

    fig,ax = plt.subplots()
    ax.imshow(image)

    for idx,(x1,y1,x2,y2,cls_id) in enumerate(bboxes):
        x,y = x1,y1
        w,h = (x2-x1),(y2-y1)
        # 创建一个矩形对象（bounding box）
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none', linestyle='-')
        # 画框框
        ax.add_patch(rect)
        # 画标签
        ax.text(x,y,class_names[cls_id],color = "black",fontsize = 14,weight='bold',bbox=dict(facecolor='red', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3'))
    img_name = train_dataset.get_image_id(img_idx)
    plt.savefig(osp.join(test_save_path,img_name + ".jpg"))
    plt.show()


def test_data_set():
    # img_idx: 90
    # img_idx: 328
    # img_idx: 685
    # img_idx: 854
    train_dataset, val_dataset = get_dataset(args, class_names)
    train_length = len(train_dataset)
    print(f"train_length: {train_length}")
    for img_idx in range(train_length):
        image,bboxes = train_dataset.parse_annotation(img_idx)
        cls_list = [bbox[4] for bbox in bboxes]
        cls_set = set(cls_list)
        if len(cls_set) > 3 and len(cls_set) < len(bboxes):
            print(f"img_idx: {img_idx}")

def draw_heatmap():
    train_dataset, val_dataset = get_dataset(args, class_names)
    img_idx = 90
    image, bboxes = train_dataset.parse_annotation(img_idx)

    clses = list(set([bbox[4] for bbox in bboxes]))
    print(f"clses: {len(clses)},{clses}")
    image, batch_hm, batch_wh, batch_offset, batch_offset_mask = train_dataset[img_idx]
    print(f"image: {image.shape}")
    print(f"batch_hm: {batch_hm.shape}")
    print(f"batch_wh: {batch_wh.shape}")
    print(f"batch_offset: {batch_offset.shape}")
    print(f"batch_offset_mask: {batch_offset_mask.shape}")

    # region heatmap
    hm_list = [batch_hm[...,cls] for cls in clses]

    # 创建一个 3x3 的子图布局 (最多 9 个子图)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()

    for i,(cls,hm_image) in enumerate(zip(clses,hm_list)):
        axes[i].imshow(hm_image,cmap = 'gray',vmin = 0,vmax=1)
        axes[i].axis('off')
        axes[i].set_title(f"{class_names[cls]}", fontsize=10)

    # 关闭没有图像的子图
    for i in range(len(hm_list), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    img_name = train_dataset.get_image_id(img_idx) +"_heatmap"
    plt.savefig(osp.join(test_save_path,img_name + ".jpg"))
    plt.show()
    # endregion
    # region batch_wh

def draw_wh_offset():
    train_dataset, val_dataset = get_dataset(args, class_names)
    img_idx = 90
    image, bboxes = train_dataset.parse_annotation(img_idx)

    clses = list(set([bbox[4] for bbox in bboxes]))
    print(f"clses: {len(clses)},{clses}")
    batch_hm,bbox_center_wh,batch_offset = train_dataset.debug_sparse_infos(img_idx)
    print(f"image: {image.shape}")
    print(f"batch_hm: {batch_hm.shape}")
    print(f"image_center_wh: {len(bbox_center_wh)},image_center_offset: {batch_offset.shape}")

    height, width, channels = batch_hm.shape
    image_wh = np.zeros((height, width, 3))  # 生成一个全黑的 RGB 图像
    image_offset = np.zeros((height, width, 3))  # 生成一个全黑的 RGB 图像

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_wh)
    axes[0].set_title("image_wh")  # 给第一个子图添加标题
    axes[0].axis('off')  # 隐藏坐标轴
    for center_wh in bbox_center_wh:
        cw,ch,w,h = center_wh
        center = (cw,ch)
        left = (cw - w / 2.,ch)
        right = (cw + w / 2.,ch)
        up = (cw,ch - h / 2.)
        down = (cw,ch + h / 2.)

        axes[0].annotate('', xy=left, xytext=center,
                      arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2))
        axes[0].annotate('', xy=right, xytext=center,
                      arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2))
        axes[0].annotate('', xy=up, xytext=center,
                      arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2))
        axes[0].annotate('', xy=down, xytext=center,
                      arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle="->", lw=2))

    axes[1].imshow(image_offset)
    axes[1].set_title("image_offset_w")  # 给第一个子图添加标题
    axes[1].axis('off')  # 隐藏坐标轴
    axes[1].imshow(batch_offset[...,0])

    axes[2].imshow(image_offset)
    axes[2].set_title("image_offset_h")  # 给第一个子图添加标题
    axes[2].axis('off')  # 隐藏坐标轴
    axes[2].imshow(batch_offset[...,1])

    # 显示图像
    plt.tight_layout()  # 自动调整子图间距
    img_name = train_dataset.get_image_id(img_idx) +"_wh_offset"
    plt.savefig(osp.join(test_save_path,img_name + ".jpg"))
    plt.show()

def test():
    np.random.seed(42)  # 设置随机种子，确保每次运行结果一致

    # 创建一个100x100的图像，随机生成每个像素的宽高 (w, h)
    image_wh = np.random.randint(1, 10, size=(100, 100, 2))  # 每个像素的宽高范围在 1 到 10 之间

    # 创建一个绘图窗口
    fig, ax = plt.subplots(figsize=(10, 10))

    # 设置网格（背景图）
    ax.set_xticks(np.arange(0, 101, 1))
    ax.set_yticks(np.arange(0, 101, 1))
    ax.grid(which='both')

    # 绘制每个格子的箭头
    for i in range(100):
        for j in range(100):
            w, h = image_wh[i, j]

            if w > 0 and h > 0:
                # 计算中心点
                x_center, y_center = j + 0.5, i + 0.5  # 每个像素的中心
                # 绘制左右箭头，长度为 w/2，方向为水平方向
                ax.arrow(x_center, y_center, w / 2, 0, head_width=0.5, head_length=1, fc='r', ec='r')
                ax.arrow(x_center, y_center, -w / 2, 0, head_width=0.5, head_length=1, fc='r', ec='r')
                # 绘制上下箭头，长度为 h/2，方向为垂直方向
                ax.arrow(x_center, y_center, 0, h / 2, head_width=0.5, head_length=1, fc='b', ec='b')
                ax.arrow(x_center, y_center, 0, -h / 2, head_width=0.5, head_length=1, fc='b', ec='b')

    # 设置坐标轴范围与图像大小
    ax.set_xlim(0, 100)
    ax.set_ylim(100, 0)  # y轴是倒的，和图像坐标一致
    ax.set_aspect('equal')

    # 隐藏坐标轴
    ax.axis('off')

    # 显示图像
    plt.title('Visualization of Object Width and Height (w, h) with Arrows')
    plt.show()

if __name__ == '__main__':
    # test_train()
    # transform_image_bboxes()
    # ann_to_label()
    # draw_bboxes()
    # test_data_set()
    # draw_heatmap()
    # test()
    draw_wh_offset()