import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import DataListLoader
from tensorboardX import SummaryWriter

from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss
import dataset
from dataset import Normalize
from train import train_epoch
from test import test_epoch
from utils.config import cfg
from utils.util import create_folder

import os
import logging
import argparse
from datetime import datetime

# Argument parse
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument('--cfg', default='configs/train/yumi.yaml', type=str, help='Path to configuration file')
args = parser.parse_args()

# Configurations parse
cfg.merge_from_file(args.cfg)
cfg.freeze()
print(cfg)

# Create folder
create_folder(cfg.OTHERS.SAVE)
create_folder(cfg.OTHERS.LOG)
create_folder(cfg.OTHERS.SUMMARY)

# Create logger & tensorboard writer
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.FileHandler(os.path.join(cfg.OTHERS.LOG, "{:%Y-%m-%d_%H-%M-%S}.log".format(datetime.now()))), logging.StreamHandler()])
logger = logging.getLogger()
writer = SummaryWriter(os.path.join(cfg.OTHERS.SUMMARY, "{:%Y-%m-%d_%H-%M-%S}".format(datetime.now())))

# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    # Load data
    pre_transform = transforms.Compose([Normalize()])
    train_set = getattr(dataset, cfg.DATASET.TRAIN.SOURCE_NAME)(root=cfg.DATASET.TRAIN.SOURCE_PATH, pre_transform=pre_transform)
    train_loader = DataListLoader(train_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    train_target = sorted([target for target in getattr(dataset, cfg.DATASET.TRAIN.TARGET_NAME)(root=cfg.DATASET.TRAIN.TARGET_PATH)], key=lambda target : target.skeleton_type)
    test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(root=cfg.DATASET.TEST.SOURCE_PATH, pre_transform=pre_transform)
    test_loader = DataListLoader(test_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)

    # Create model
    model = getattr(model, cfg.MODEL.NAME)().to(device)

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    # Create loss criterion
    # end effector loss
    ee_criterion = nn.MSELoss() if cfg.LOSS.EE else None
    # vector similarity loss
    vec_criterion = nn.MSELoss() if cfg.LOSS.VEC else None
    # collision loss
    col_criterion = CollisionLoss(cfg.LOSS.COL_THRESHOLD) if cfg.LOSS.COL else None
    # joint limit loss
    lim_criterion = JointLimitLoss() if cfg.LOSS.LIM else None
    # end effector orientation loss
    ori_criterion = nn.MSELoss() if cfg.LOSS.ORI else None
    # finger similarity loss
    fin_criterion = nn.MSELoss() if cfg.LOSS.FIN else None
    # regularization loss
    reg_criterion = RegLoss() if cfg.LOSS.REG else None

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.HYPER.LEARNING_RATE)

    best_loss = float('Inf')

    for epoch in range(cfg.HYPER.EPOCHS):
        # Start training
        train_loss = train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,
                                 optimizer, train_loader, train_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device)

        # Start testing
        test_loss = test_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,
                               test_loader, test_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device)

        # Save model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(cfg.OTHERS.SAVE, "best_model_epoch_{:04d}.pth".format(epoch)))
            logger.info("Epoch {} Model Saved".format(epoch+1).center(60, '-'))
