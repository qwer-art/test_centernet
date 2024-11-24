import sys
import os.path as osp

from core.loss import focal_loss,l1_loss
from net import CenterNet

project_path = osp.abspath(osp.join(osp.dirname(__file__),".."))
sys.path.append(project_path)

from core.helper import get_model,get_dataset
from tools.args import args, dev, class_names
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_train_loss():
    # dataset
    train_dataset, val_dataset = get_dataset(args, class_names)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=1)
    img_idx = 90

    # model
    pretrain_weight_path = "/home/zyt/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"
    model = CenterNet("resnet50",num_classes=len(class_names))
    model_state_dict = model.state_dict()
    pretrain_state_dict = torch.load(args.pretrain_weight_path)
    for k, v in pretrain_state_dict.items():
        centernet_k = "backbone." + k
        if centernet_k in model_state_dict.keys():
            model_state_dict[centernet_k] = v
    model.load_state_dict(model_state_dict)

    criterion = nn.CrossEntropyLoss()
    alpha = 1
    gamma = 2
    reduction = 'mean'

    ### model train result
    for batch_idx, img_info in enumerate(train_loader):
        if batch_idx != img_idx:
            continue
        images, hms_true, whs_true, offsets_true, offset_masks_true = img_info
        training_output = model(images, mode='train', ground_truth_data=(hms_true,
                                                                         whs_true,
                                                                         offsets_true,
                                                                         offset_masks_true))
        hms_pred, whs_pred, offsets_pred, loss, c_loss, wh_loss, off_loss, hms_true = training_output
        print(f"hms_pred: {hms_pred.shape},whs_pred: {whs_pred.shape},offsets_pred: {offsets_pred.shape},")
        print(f"loss: {loss},c_loss: {c_loss},wh_loss: {wh_loss},off_loss: {off_loss}")

    ### test train result
    for batch_idx, img_info in enumerate(train_loader):
        if batch_idx != img_idx:
            continue
        print(f"img_idx: {batch_idx}")
        print("================== True ==================")
        images, hms_true, whs_true, offsets_true, offset_masks_true = img_info
        hms_true = hms_true.permute(0, 3, 1, 2)
        whs_true = whs_true.permute(0, 3, 1, 2)
        offsets_true = offsets_true.permute(0, 3, 1, 2)
        # offset_masks_true = offset_masks_true.permute(0,3,1,2)

        images = images.to(torch.float32)
        print(f"images: {images.shape},{images.dtype}")
        print(f"hms_true: {hms_true.shape},{hms_true.dtype}")
        print(f"whs_true: {whs_true.shape},{whs_true.dtype}")
        print(f"offsets_true: {offsets_true.shape},{offsets_true.dtype}")
        print(f"offset_masks_true: {offset_masks_true.shape},{offset_masks_true.dtype}")


        print("================== Pred ==================")
        hms_pred, whs_pred, offsets_pred = model.pred(images)
        print(f"hms_pred: {hms_pred.shape}")
        print(f"whs_pred: {whs_pred.shape}")
        print(f"offsets_pred: {offsets_pred.shape}")

        c_loss = focal_loss(hms_pred, hms_true)
        wh_loss = 0.1 * l1_loss(whs_pred, whs_true, offset_masks_true)
        off_loss = l1_loss(offsets_pred, offsets_true, offset_masks_true)
        loss = c_loss + wh_loss + off_loss
        print(f"c_loss: {c_loss},wh_loss: {wh_loss},off_loss: {off_loss},loss: {loss}")


if __name__ == '__main__':
    get_train_loss()