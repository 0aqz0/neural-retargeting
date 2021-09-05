import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss
import time

def train_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion, optimizer, dataloader, target_skeleton, epoch, logger, log_interval, writer, device, z_all=None, ang_all=None):
    logger.info("Training Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.train()
    all_losses = []
    ee_losses = []
    vec_losses = []
    col_losses = []
    lim_losses = []
    ori_losses = []
    fin_losses = []
    reg_losses = []

    for batch_idx, data_list in enumerate(dataloader):
        for target_idx, target in enumerate(target_skeleton):
            # zero gradient
            optimizer.zero_grad()

            # fetch target
            target_list = [target for data in data_list]

            # forward
            if z_all is not None:
                z = z_all[batch_idx]
                _, target_ang, target_pos, target_rot, target_global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = model.decode(z, Batch.from_data_list(target_list).to(device))
            else:
                z, target_ang, target_pos, target_rot, target_global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device))

            # calculate all loss
            loss = calculate_all_loss(data_list, target_list, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,
                                      z, target_ang, target_pos, target_rot, target_global_pos, l_hand_pos, r_hand_pos, all_losses, ee_losses, vec_losses, col_losses, lim_losses, ori_losses, fin_losses, reg_losses)

            # backward
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # optimize
            optimizer.step()

        # log
        if (batch_idx + 1) % log_interval == 0:
            logger.info("epoch {:04d} | iteration {:05d} | EE {:.6f} | Vec {:.6f} | Col {:.6f} | Lim {:.6f} | Ori {:.6f} | Fin {:.6f} | Reg {:.6f}".format(epoch+1, batch_idx+1, ee_losses[-1], vec_losses[-1], col_losses[-1], lim_losses[-1], ori_losses[-1], fin_losses[-1], reg_losses[-1]))

    # Compute average loss
    train_loss = sum(all_losses)/len(all_losses)
    ee_loss = sum(ee_losses)/len(ee_losses)
    vec_loss = sum(vec_losses)/len(vec_losses)
    col_loss = sum(col_losses)/len(col_losses)
    lim_loss = sum(lim_losses)/len(lim_losses)
    ori_loss = sum(ori_losses)/len(ori_losses)
    fin_loss = sum(fin_losses)/len(fin_losses)
    reg_loss = sum(reg_losses)/len(reg_losses)
    # Log
    writer.add_scalars('training_loss', {'train': train_loss}, epoch+1)
    writer.add_scalars('end_effector_loss', {'train': ee_loss}, epoch+1)
    writer.add_scalars('vector_loss', {'train': vec_loss}, epoch+1)
    writer.add_scalars('collision_loss', {'train': col_loss}, epoch+1)
    writer.add_scalars('joint_limit_loss', {'train': lim_loss}, epoch+1)
    writer.add_scalars('orientation_loss', {'train': ori_loss}, epoch+1)
    writer.add_scalars('finger_loss', {'train': fin_loss}, epoch+1)
    writer.add_scalars('regularization_loss', {'train': reg_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Training Time {:.2f} s | Avg Training Loss {:.6f} | Avg EE Loss {:.6f} | Avg Vec Loss {:.6f} | Avg Col Loss {:.6f} | Avg Lim Loss {:.6f} | Avg Ori Loss {:.6f} | Avg Fin Loss {:.6f} | Avg Reg Loss {:.6f}".format(epoch+1, end_time-start_time, train_loss, ee_loss, vec_loss, col_loss, lim_loss, ori_loss, fin_loss, reg_loss))

    return train_loss
