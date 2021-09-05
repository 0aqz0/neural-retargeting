import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as transforms
from torch_geometric.data import Batch, DataListLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import h5py
import argparse
import logging
import time
import os
import copy
from datetime import datetime

import dataset
from dataset import Normalize, parse_h5, parse_h5_hand, parse_all
from models import model
from models.loss import CollisionLoss, JointLimitLoss, RegLoss
from train import train_epoch
from utils.config import cfg
from utils.util import create_folder

# Argument parse
parser = argparse.ArgumentParser(description='Inference with trained model')
parser.add_argument('--cfg', default='configs/inference/yumi.yaml', type=str, help='Path to configuration file')
args = parser.parse_args()

# Configurations parse
cfg.merge_from_file(args.cfg)
cfg.freeze()
print(cfg)

# Create folder
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
    if cfg.INFERENCE.MOTION.KEY:
        # inference single key
        print('Inference single key {}'.format(cfg.INFERENCE.MOTION.KEY))
        # test_data, l_hand_angle, r_hand_angle = parse_h5(filename=cfg.INFERENCE.MOTION.SOURCE, selected_key=cfg.INFERENCE.MOTION.KEY)
        # test_data = parse_h5_hand(filename=cfg.INFERENCE.MOTION.SOURCE, selected_key=cfg.INFERENCE.MOTION.KEY)
        test_data = parse_all(filename=cfg.INFERENCE.MOTION.SOURCE, selected_key=cfg.INFERENCE.MOTION.KEY)
        test_data = [pre_transform(data) for data in test_data]
        indices = [idx for idx in range(0, len(test_data), cfg.HYPER.BATCH_SIZE)]
        test_loader = [test_data]#[test_data[idx: idx+cfg.HYPER.BATCH_SIZE] for idx in indices]
        hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, 'source.h5'), 'w')
        g1 = hf.create_group('group1')
        source_pos = torch.stack([data.pos for data in test_data], dim=0)
        g1.create_dataset('l_joint_pos', data=source_pos[:, :3])
        g1.create_dataset('r_joint_pos', data=source_pos[:, 3:])
        hf.close()
        print('Source H5 file saved!')
    else:
        # inference all
        print('Inference all')
        test_set = getattr(dataset, cfg.DATASET.TEST.SOURCE_NAME)(root=cfg.DATASET.TEST.SOURCE_PATH, pre_transform=pre_transform)
        test_loader = DataListLoader(test_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
    test_target = sorted([target for target in getattr(dataset, cfg.DATASET.TEST.TARGET_NAME)(root=cfg.DATASET.TEST.TARGET_PATH)], key=lambda target : target.skeleton_type)

    # Create model
    model = getattr(model, cfg.MODEL.NAME)().to(device)

    # Load checkpoint
    if cfg.MODEL.CHECKPOINT is not None:
        model.load_state_dict(torch.load(cfg.MODEL.CHECKPOINT))

    # training set z mean & std
    # train_set = getattr(dataset, "SignAll")(root="./data/source/sign-all/train", pre_transform=pre_transform)
    # train_loader = DataListLoader(train_set, batch_size=cfg.HYPER.BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    # train_target = sorted([target for target in getattr(dataset, "YumiAll")(root="./data/target/yumi-all")], key=lambda target : target.skeleton_type)
    # model.eval()
    # z_train = []
    # for batch_idx, data_list in enumerate(train_loader):
    #     for target_idx, target in enumerate(train_target):
    #         # fetch target
    #         target_list = [target for data in data_list]
    #         # forward
    #         z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
    #         z_train.append(z)
    # z_train = torch.cat(z_train, dim=0)
    # mean = z_train.mean(0)
    # std = z_train.std(0)
    # print(z_train.shape, mean.shape, std.shape)
    # print(mean, std)

    # store initial z
    encode_start_time = time.time()
    model.eval()
    z_all = []
    for batch_idx, data_list in enumerate(test_loader):
        for target_idx, target in enumerate(test_target):
            # forward
            z = model.encode(Batch.from_data_list(data_list).to(device)).detach()
            # target_batch = Batch.from_data_list([target for data in data_list])
            # target_nodes = target_batch.x.size(0)+2*target_batch.hand_x.size(0)
            # z = torch.empty(target_nodes, 64).normal_(mean=0, std=0.1).to(device)
            # z = torch.zeros(target_nodes, 64).to(device)
            # z = torch.stack([torch.normal(mean=mean, std=std) for _ in range(target_nodes)], dim=0).to(device)
            z.requires_grad = True
            z_all.append(z)
    encode_end_time = time.time()
    print('encode time {} ms'.format((encode_end_time - encode_start_time)*1000))
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
    optimizer = optim.Adam(z_all, lr=cfg.HYPER.LEARNING_RATE)

    best_loss = float('Inf')
    best_z_all = copy.deepcopy(z_all)
    best_cnt = 0
    start_time = time.time()

    # latent optimization
    decode_start_time = time.time()
    for epoch in range(cfg.HYPER.EPOCHS):
        train_loss = train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion, optimizer, test_loader, test_target, epoch, logger, cfg.OTHERS.LOG_INTERVAL, writer, device, z_all)
        if cfg.INFERENCE.MOTION.KEY:
            # Save model
            if train_loss > best_loss:
                best_cnt += 1
            else:
                best_cnt = 0
                best_loss = train_loss
                best_z_all = copy.deepcopy(z_all)
            if best_cnt == 5:
                logger.info("Interation Finished")
                break
            print(best_cnt)
    decode_end_time = time.time()
    print('decode time {} ms'.format((decode_end_time - decode_start_time)*1000))
    if cfg.INFERENCE.MOTION.KEY:
        # store final results
        model.eval()
        pos_all = []
        ang_all = []
        l_hand_ang_all = []
        r_hand_ang_all = []
        for batch_idx, data_list in enumerate(test_loader):
            for target_idx, target in enumerate(test_target):
                # fetch target
                target_list = [target for data in data_list]
                # fetch z
                z = best_z_all[batch_idx]
                # forward
                _, target_ang, _, _, target_global_pos, l_hand_ang, _, r_hand_ang, _ = model.decode(z, Batch.from_data_list(target_list).to(z.device))

                if target_global_pos is not None and target_ang is not None:
                    pos_all.append(target_global_pos)
                    ang_all.append(target_ang)
                if l_hand_ang is not None and r_hand_ang is not None:
                    l_hand_ang_all.append(l_hand_ang)
                    r_hand_ang_all.append(r_hand_ang)
        
        if cfg.INFERENCE.H5.BOOL:
            hf = h5py.File(os.path.join(cfg.INFERENCE.H5.PATH, 'inference.h5'), 'w')
            g1 = hf.create_group('group1')
            if pos_all and ang_all:
                pos = torch.cat(pos_all, dim=0).view(len(test_data), -1, 3).detach().cpu().numpy() # [T, joint_num, xyz]
                ang = torch.cat(ang_all, dim=0).view(len(test_data), -1).detach().cpu().numpy()
                g1.create_dataset('l_joint_pos', data=pos[:, :7])
                g1.create_dataset('r_joint_pos', data=pos[:, 7:])
                g1.create_dataset('l_joint_angle', data=ang[:, :7])
                g1.create_dataset('r_joint_angle', data=ang[:, 7:])
                # g1.create_dataset('l_glove_angle', data=l_hand_angle)
                # g1.create_dataset('r_glove_angle', data=r_hand_angle)
            if l_hand_ang_all and r_hand_ang_all:
                l_hand_angle = torch.cat(l_hand_ang_all, dim=0).view(len(test_data), -1).detach().cpu().numpy()
                r_hand_angle = torch.cat(r_hand_ang_all, dim=0).view(len(test_data), -1).detach().cpu().numpy()
                # remove zeros
                l_hand_angle = np.concatenate([l_hand_angle[:,1:3],l_hand_angle[:,4:6],l_hand_angle[:,7:9],l_hand_angle[:,10:12],l_hand_angle[:,13:17]], axis=1)
                r_hand_angle = np.concatenate([r_hand_angle[:,1:3],r_hand_angle[:,4:6],r_hand_angle[:,7:9],r_hand_angle[:,10:12],r_hand_angle[:,13:17]], axis=1)
                g1.create_dataset('l_glove_angle', data=l_hand_angle)
                g1.create_dataset('r_glove_angle', data=r_hand_angle)
            hf.close()
            print('Target H5 file saved!')
