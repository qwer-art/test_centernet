import sys
import os.path as osp

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from net.centernet import CenterNet
from core.helper import get_model,remove_dir_and_create_dir,get_dataset
from util import *
from tools.train import CosineAnnealingWarmupRestarts
from tools.args import args, dev, class_names
from core.dataset.utils import image_resize, preprocess_input, gaussian_radius, draw_gaussian

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



if __name__ == '__main__':
    # test_train()
    transform_image_bboxes()
