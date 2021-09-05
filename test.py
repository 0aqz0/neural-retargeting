import torch
from torch_geometric.data import Batch
from models.loss import calculate_all_loss
import time

def test_epoch(model, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion, dataloader, target_skeleton, epoch, logger, log_interval, writer, device):
    logger.info("Testing Epoch {}".format(epoch+1).center(60, '-'))
    start_time = time.time()

    model.eval()
    all_losses = []
    ee_losses = []
    vec_losses = []
    col_losses = []
    lim_losses = []
    ori_losses = []
    fin_losses = []
    reg_losses = []

    with torch.no_grad():
        for batch_idx, data_list in enumerate(dataloader):
            for target_idx, target in enumerate(target_skeleton):
                # fetch target
                target_list = [target for data in data_list]

                # forward
                z, target_ang, target_pos, target_rot, target_global_pos, l_hand_ang, l_hand_pos, r_hand_ang, r_hand_pos = model(Batch.from_data_list(data_list).to(device), Batch.from_data_list(target_list).to(device))

                # calculate all loss
                loss = calculate_all_loss(data_list, target_list, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,
                                          z, target_ang, target_pos, target_rot, target_global_pos, l_hand_pos, r_hand_pos, all_losses, ee_losses, vec_losses, col_losses, lim_losses, ori_losses, fin_losses, reg_losses)

    # Compute average loss
    test_loss = sum(all_losses)/len(all_losses)
    ee_loss = sum(ee_losses)/len(ee_losses)
    vec_loss = sum(vec_losses)/len(vec_losses)
    col_loss = sum(col_losses)/len(col_losses)
    lim_loss = sum(lim_losses)/len(lim_losses)
    ori_loss = sum(ori_losses)/len(ori_losses)
    fin_loss = sum(fin_losses)/len(fin_losses)
    reg_loss = sum(reg_losses)/len(reg_losses)
    # Log
    writer.add_scalars('testing_loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('end_effector_loss', {'test': ee_loss}, epoch+1)
    writer.add_scalars('vector_loss', {'test': vec_loss}, epoch+1)
    writer.add_scalars('collision_loss', {'test': col_loss}, epoch+1)
    writer.add_scalars('joint_limit_loss', {'test': lim_loss}, epoch+1)
    writer.add_scalars('orientation_loss', {'test': ori_loss}, epoch+1)
    writer.add_scalars('finger_loss', {'test': fin_loss}, epoch+1)
    writer.add_scalars('regularization_loss', {'test': reg_loss}, epoch+1)
    end_time = time.time()
    logger.info("Epoch {:04d} | Testing Time {:.2f} s | Avg Testing Loss {:.6f} | Avg EE Loss {:.6f} | Avg Vec Loss {:.6f} | Avg Col Loss {:.6f} | Avg Lim Loss {:.6f} | Avg Ori Loss {:.6f} | Avg Fin Loss {:.6f} | Avg Reg Loss {:.6f}".format(epoch+1, end_time-start_time, test_loss, ee_loss, vec_loss, col_loss, lim_loss, ori_loss, fin_loss, reg_loss))

    return test_loss
