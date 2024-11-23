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

def test_train():
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "weights"), is_remove=True)
    remove_dir_and_create_dir(os.path.join(args.logs_dir, "summary"), is_remove=True)

    model = get_model(args, dev)
    train_dataset, val_dataset = get_dataset(args, class_names)

    writer = SummaryWriter(os.path.join(args.logs_dir, "summary"))

    freeze_step = len(train_dataset) // args.freeze_batch_size
    unfreeze_step = len(train_dataset) // args.unfreeze_batch_size
    print(f"freezee_step: {freeze_step},unfreeze_step: {unfreeze_step}")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, args.learn_rate_init)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                              first_cycle_steps=args.freeze_epochs * freeze_step + args.unfreeze_epochs * unfreeze_step,
                                              max_lr=args.learn_rate_init,
                                              min_lr=args.learn_rate_end,
                                              warmup_steps=args.warmup_epochs * freeze_step)


if __name__ == '__main__':
    step = 0
    test_train()