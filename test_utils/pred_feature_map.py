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
    img_idx = 100
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

    plt.show()

if __name__ == '__main__':
    # test_train()
    # transform_image_bboxes()
    # ann_to_label()
    draw_bboxes()