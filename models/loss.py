import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry.conversions import quaternion_to_rotation_matrix

"""
Calculate All Loss
"""
def calculate_all_loss(data_list, target_list, ee_criterion, vec_criterion, col_criterion, lim_criterion, ori_criterion, fin_criterion, reg_criterion,
                       z, target_ang, target_pos, target_rot, target_global_pos, l_hand_pos, r_hand_pos, all_losses=[], ee_losses=[], vec_losses=[], col_losses=[], lim_losses=[], ori_losses=[], fin_losses=[], reg_losses=[]):
    # end effector loss
    if ee_criterion:
        ee_loss = calculate_ee_loss(data_list, target_list, target_pos, ee_criterion)*1000
        ee_losses.append(ee_loss.item())
    else:
        ee_loss = 0
        ee_losses.append(0)

    # vector loss
    if vec_criterion:
        vec_loss = calculate_vec_loss(data_list, target_list, target_pos, vec_criterion)*100
        vec_losses.append(vec_loss.item())
    else:
        vec_loss = 0
        vec_losses.append(0)

    # collision loss
    if col_criterion:
        col_loss = col_criterion(target_global_pos.view(len(target_list), -1, 3), target_list[0].edge_index,
                                 target_rot.view(len(target_list), -1, 9), target_list[0].ee_mask)*1000
        col_losses.append(col_loss.item())
    else:
        col_loss = 0
        col_losses.append(0)

    # joint limit loss
    if lim_criterion:
        lim_loss = calculate_lim_loss(target_list, target_ang, lim_criterion)*10000
        lim_losses.append(lim_loss.item())
    else:
        lim_loss = 0
        lim_losses.append(0)

    # end effector orientation loss
    if ori_criterion:
        ori_loss = calculate_ori_loss(data_list, target_list, target_rot, ori_criterion)*100
        ori_losses.append(ori_loss.item())
    else:
        ori_loss = 0
        ori_losses.append(0)

    # finger similarity loss
    if fin_criterion:
        fin_loss = calculate_fin_loss(data_list, target_list, l_hand_pos, r_hand_pos, fin_criterion)*100
        fin_losses.append(fin_loss.item())
    else:
        fin_loss = 0
        fin_losses.append(0)

    # regularization loss
    if reg_criterion:
        reg_loss = reg_criterion(z.view(len(target_list), -1, 64))
        reg_losses.append(reg_loss.item())
    else:
        reg_loss = 0
        reg_losses.append(0)

    # total loss
    loss = ee_loss + vec_loss + col_loss + lim_loss + ori_loss + fin_loss + reg_loss
    all_losses.append(loss.item())

    return loss

"""
Calculate End Effector Loss
"""
def calculate_ee_loss(data_list, target_list, target_pos, ee_criterion):
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_pos.device)
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_pos.device)
    target_ee = torch.masked_select(target_pos, target_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device), source_mask).view(-1, 3)
    # normalize
    target_root_dist = torch.cat([data.root_dist for data in target_list]).to(target_pos.device)
    source_root_dist = torch.cat([data.root_dist for data in data_list]).to(target_pos.device)
    target_ee = target_ee / torch.masked_select(target_root_dist, target_mask).unsqueeze(1)
    source_ee = source_ee / torch.masked_select(source_root_dist, source_mask).unsqueeze(1)
    ee_loss = ee_criterion(target_ee, source_ee)
    return ee_loss

"""
Calculate Vector Loss
"""
def calculate_vec_loss(data_list, target_list, target_pos, vec_criterion):
    target_sh_mask = torch.cat([data.sh_mask for data in target_list]).to(target_pos.device)
    target_el_mask = torch.cat([data.el_mask for data in target_list]).to(target_pos.device)
    target_ee_mask = torch.cat([data.ee_mask for data in target_list]).to(target_pos.device)
    source_sh_mask = torch.cat([data.sh_mask for data in data_list]).to(target_pos.device)
    source_el_mask = torch.cat([data.el_mask for data in data_list]).to(target_pos.device)
    source_ee_mask = torch.cat([data.ee_mask for data in data_list]).to(target_pos.device)
    target_sh = torch.masked_select(target_pos, target_sh_mask).view(-1, 3)
    target_el = torch.masked_select(target_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(target_pos, target_ee_mask).view(-1, 3)
    source_sh = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device), source_sh_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device), source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.pos for data in data_list]).to(target_pos.device), source_ee_mask).view(-1, 3)
    # print(target_sh.shape, target_el.shape, target_ee.shape, source_sh.shape, source_el.shape, source_ee.shape)
    target_vector1 = target_el - target_sh
    target_vector2 = target_ee - target_el
    source_vector1 = source_el - source_sh
    source_vector2 = source_ee - source_el
    # print(target_vector1.shape, target_vector2.shape, source_vector1.shape, source_vector2.shape, (target_vector1*source_vector1).sum(-1).shape)
    # normalize
    target_shoulder_dist = torch.cat([data.shoulder_dist for data in target_list]).to(target_pos.device)
    target_elbow_dist = torch.cat([data.elbow_dist for data in target_list]).to(target_pos.device)/2
    source_shoulder_dist = torch.cat([data.shoulder_dist for data in data_list]).to(target_pos.device)
    source_elbow_dist = torch.cat([data.elbow_dist for data in data_list]).to(target_pos.device)/2
    normalize_target_vector1 = target_vector1 / torch.masked_select(target_shoulder_dist, target_el_mask).unsqueeze(1)
    normalize_source_vector1 = source_vector1 / torch.masked_select(source_shoulder_dist, source_el_mask).unsqueeze(1)
    vector1_loss = vec_criterion(normalize_target_vector1, normalize_source_vector1)
    normalize_target_vector2 = target_vector2 / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector2 = source_vector2 / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    vector2_loss = vec_criterion(normalize_target_vector2, normalize_source_vector2)
    vec_loss = vector2_loss#(vector1_loss + vector2_loss)*100
    return vec_loss

"""
Calculate Joint Limit Loss
"""
def calculate_lim_loss(target_list, target_ang, lim_criterion):
    target_lower = torch.cat([data.lower for data in target_list]).to(target_ang.device)
    target_upper = torch.cat([data.upper for data in target_list]).to(target_ang.device)
    lim_loss = lim_criterion(target_ang, target_lower, target_upper)
    return lim_loss

"""
Calculate Orientation Loss
"""
def calculate_ori_loss(data_list, target_list, target_rot, ori_criterion):
    target_mask = torch.cat([data.ee_mask for data in target_list]).to(target_rot.device)
    source_mask = torch.cat([data.ee_mask for data in data_list]).to(target_rot.device)
    target_rot = target_rot.view(-1, 9)
    source_rot = quaternion_to_rotation_matrix(torch.cat([data.q for data in data_list]).to(target_rot.device)).view(-1, 9)
    target_q = torch.masked_select(target_rot, target_mask)
    source_q = torch.masked_select(source_rot, source_mask)
    ori_loss = ori_criterion(target_q, source_q)
    return ori_loss

"""
Calculate Finger Similarity Loss
"""
def calculate_fin_loss(data_list, target_list, l_hand_pos, r_hand_pos, ee_criterion):
    # left hand
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(l_hand_pos.device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(l_hand_pos.device)
    source_el_mask = torch.cat([data.l_hand_el_mask for data in data_list]).to(l_hand_pos.device)
    source_ee_mask = torch.cat([data.l_hand_ee_mask for data in data_list]).to(l_hand_pos.device)
    target_el = torch.masked_select(l_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(l_hand_pos, target_ee_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.l_hand_pos for data in data_list]).to(l_hand_pos.device), source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.l_hand_pos for data in data_list]).to(l_hand_pos.device), source_ee_mask).view(-1, 3)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(l_hand_pos.device)
    source_elbow_dist = torch.cat([data.l_hand_elbow_dist for data in data_list]).to(l_hand_pos.device)
    normalize_target_vector = target_vector / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector = source_vector / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    l_fin_loss = ee_criterion(normalize_target_vector, normalize_source_vector)

    # right hand
    target_el_mask = torch.cat([data.hand_el_mask for data in target_list]).to(r_hand_pos.device)
    target_ee_mask = torch.cat([data.hand_ee_mask for data in target_list]).to(r_hand_pos.device)
    source_el_mask = torch.cat([data.r_hand_el_mask for data in data_list]).to(r_hand_pos.device)
    source_ee_mask = torch.cat([data.r_hand_ee_mask for data in data_list]).to(r_hand_pos.device)
    target_el = torch.masked_select(r_hand_pos, target_el_mask).view(-1, 3)
    target_ee = torch.masked_select(r_hand_pos, target_ee_mask).view(-1, 3)
    source_el = torch.masked_select(torch.cat([data.r_hand_pos for data in data_list]).to(r_hand_pos.device), source_el_mask).view(-1, 3)
    source_ee = torch.masked_select(torch.cat([data.r_hand_pos for data in data_list]).to(r_hand_pos.device), source_ee_mask).view(-1, 3)
    target_vector = target_ee - target_el
    source_vector = source_ee - source_el
    # normalize
    target_elbow_dist = torch.cat([data.hand_elbow_dist for data in target_list]).to(r_hand_pos.device)
    source_elbow_dist = torch.cat([data.r_hand_elbow_dist for data in data_list]).to(r_hand_pos.device)
    normalize_target_vector = target_vector / torch.masked_select(target_elbow_dist, target_ee_mask).unsqueeze(1)
    normalize_source_vector = source_vector / torch.masked_select(source_elbow_dist, source_ee_mask).unsqueeze(1)
    r_fin_loss = ee_criterion(normalize_target_vector, normalize_source_vector)

    fin_loss = (l_fin_loss + r_fin_loss)/2
    return fin_loss

"""
Collision Loss
"""
class CollisionLoss(nn.Module):
    def __init__(self, threshold, mode='capsule-capsule'):
        super(CollisionLoss, self).__init__()
        self.threshold = threshold
        self.mode = mode
        
    def forward(self, pos, edge_index, rot, ee_mask):
        """
        Keyword arguments:
        pos -- joint positions [batch_size, num_nodes, 3]
        edge_index -- edge index [2, num_edges]
        """
        batch_size = pos.shape[0]
        num_nodes = pos.shape[1]
        num_edges = edge_index.shape[1]

        # sphere-sphere detection
        if self.mode == 'sphere-sphere':
            l_sphere = pos[:, :num_nodes//2, :]
            r_sphere = pos[:, num_nodes//2:, :]
            l_sphere = l_sphere.unsqueeze(1).expand(batch_size, num_nodes//2, num_nodes//2, 3)
            r_sphere = r_sphere.unsqueeze(2).expand(batch_size, num_nodes//2, num_nodes//2, 3)
            dist_square = torch.sum(torch.pow(l_sphere - r_sphere, 2), dim=-1)
            mask = (dist_square < self.threshold**2) & (dist_square > 0)
            loss = torch.sum(torch.exp(-1*torch.masked_select(dist_square, mask)))/batch_size

        # sphere-capsule detection
        if self.mode == 'sphere-capsule':
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # print(edge_index.shape, p0.shape, p1.shape)

            # left sphere & right capsule
            l_sphere = pos[:, :num_nodes//2, :]
            r_capsule_p0 = p0[:, num_edges//2:, :]
            r_capsule_p1 = p1[:, num_edges//2:, :]
            dist_square_1 = self.sphere_capsule_dist_square(l_sphere, r_capsule_p0, r_capsule_p1, batch_size, num_nodes, num_edges)

            # left capsule & right sphere
            r_sphere = pos[:, num_nodes//2:, :]
            l_capsule_p0 = p0[:, :num_edges//2, :]
            l_capsule_p1 = p1[:, :num_edges//2, :]
            dist_square_2 = self.sphere_capsule_dist_square(r_sphere, l_capsule_p0, l_capsule_p1, batch_size, num_nodes, num_edges)

            # calculate loss
            dist_square = torch.cat([dist_square_1, dist_square_2])
            mask = (dist_square < self.threshold**2) & (dist_square > 0)
            loss = torch.sum(torch.exp(-1*torch.masked_select(dist_square, mask)))/batch_size

        # capsule-capsule detection
        if self.mode == 'capsule-capsule':
            # capsule p0 & p1
            p0 = pos[:, edge_index[0], :]
            p1 = pos[:, edge_index[1], :]
            # left capsule
            l_capsule_p0 = p0[:, :num_edges//2, :]
            l_capsule_p1 = p1[:, :num_edges//2, :]
            # right capsule
            r_capsule_p0 = p0[:, num_edges//2:, :]
            r_capsule_p1 = p1[:, num_edges//2:, :]
            # add capsule for left hand & right hand(for yumi)
            ee_pos = torch.masked_select(pos, ee_mask.to(pos.device)).view(-1, 3)
            ee_rot = torch.masked_select(rot, ee_mask.to(pos.device)).view(-1, 3, 3)
            offset = torch.Tensor([[[0], [0], [0.2]]]).repeat(ee_rot.size(0), 1, 1).to(pos.device)
            hand_pos = torch.bmm(ee_rot, offset).squeeze() + ee_pos
            l_ee_pos = ee_pos[::2, :].unsqueeze(1)
            l_hand_pos = hand_pos[::2, :].unsqueeze(1)
            r_ee_pos = ee_pos[1::2, :].unsqueeze(1)
            r_hand_pos = hand_pos[1::2, :].unsqueeze(1)
            l_capsule_p0 = torch.cat([l_capsule_p0, l_ee_pos], dim=1)
            l_capsule_p1 = torch.cat([l_capsule_p1, l_hand_pos], dim=1)
            r_capsule_p0 = torch.cat([r_capsule_p0, r_ee_pos], dim=1)
            r_capsule_p1 = torch.cat([r_capsule_p1, r_hand_pos], dim=1)
            num_edges += 2
            # print(l_capsule_p0.shape, l_capsule_p1.shape, r_capsule_p0.shape, r_capsule_p1.shape)
            # calculate loss
            dist_square = self.capsule_capsule_dist_square(l_capsule_p0, l_capsule_p1, r_capsule_p0, r_capsule_p1, batch_size, num_edges)
            mask = (dist_square < 0.1**2) & (dist_square > 0)
            mask[:, 6, 6] = (dist_square[:, 6, 6] < self.threshold**2) & (dist_square[:, 6, 6] > 0)
            loss = torch.sum(torch.exp(-1*torch.masked_select(dist_square, mask)))/batch_size

        return loss

    def sphere_capsule_dist_square(self, sphere, capsule_p0, capsule_p1, batch_size, num_nodes, num_edges):
        # condition 1: p0 is the closest point
        vec_p0_p1 = capsule_p1 - capsule_p0 # vector p0-p1 [batch_size, num_edges//2, 3]
        vec_p0_pr = sphere.unsqueeze(2).expand(batch_size, num_nodes//2, num_edges//2, 3) - capsule_p0.unsqueeze(1).expand(batch_size, num_nodes//2, num_edges//2, 3) # vector p0-pr [batch_size, num_nodes//2, num_edges//2, 3]
        vec_mul_p0 = torch.mul(vec_p0_p1.unsqueeze(1).expand(batch_size, num_nodes//2, num_edges//2, 3), vec_p0_pr).sum(dim=-1) # vector p0-p1 * vector p0-pr [batch_size, num_nodes//2, num_edges//2]
        dist_square_p0 = torch.masked_select(vec_p0_pr.norm(dim=-1)**2, vec_mul_p0 <= 0)
        # print(dist_square_p0.shape)

        # condition 2: p1 is the closest point
        vec_p1_p0 = capsule_p0 - capsule_p1 # vector p1-p0 [batch_size, num_edges//2, 3]
        vec_p1_pr = sphere.unsqueeze(2).expand(batch_size, num_nodes//2, num_edges//2, 3) - capsule_p1.unsqueeze(1).expand(batch_size, num_nodes//2, num_edges//2, 3) # vector p1-pr [batch_size, num_nodes//2, num_edges//2, 3]
        vec_mul_p1 = torch.mul(vec_p1_p0.unsqueeze(1).expand(batch_size, num_nodes//2, num_edges//2, 3), vec_p1_pr).sum(dim=-1) # vector p1-p0 * vector p1-pr [batch_size, num_nodes//2, num_edges//2]
        dist_square_p1 = torch.masked_select(vec_p1_pr.norm(dim=-1)**2, vec_mul_p1 <= 0)
        # print(dist_square_p1.shape)

        # condition 3: closest point in p0-p1 segement
        d = vec_mul_p0 / vec_p0_p1.norm(dim=-1).unsqueeze(1).expand(batch_size, num_nodes//2, num_edges//2) # vector p0-p1 * vector p0-pr / |vector p0-p1| [batch_size, num_nodes//2, num_edges//2]
        dist_square_middle = vec_p0_pr.norm(dim=-1)**2 - d**2 # distance square [batch_size, num_nodes//2, num_edges//2]
        dist_square_middle = torch.masked_select(dist_square_middle, (vec_mul_p0 > 0) & (vec_mul_p1 > 0))
        # print(dist_square_middle.shape)

        return torch.cat([dist_square_p0, dist_square_p1, dist_square_middle])

    def capsule_capsule_dist_square(self, capsule_p0, capsule_p1, capsule_q0, capsule_q1, batch_size, num_edges):
        # expand left capsule
        capsule_p0 = capsule_p0.unsqueeze(1).expand(batch_size, num_edges//2, num_edges//2, 3)
        capsule_p1 = capsule_p1.unsqueeze(1).expand(batch_size, num_edges//2, num_edges//2, 3)
        # expand right capsule
        capsule_q0 = capsule_q0.unsqueeze(2).expand(batch_size, num_edges//2, num_edges//2, 3)
        capsule_q1 = capsule_q1.unsqueeze(2).expand(batch_size, num_edges//2, num_edges//2, 3)
        # basic variables
        a = torch.mul(capsule_p1 - capsule_p0, capsule_p1 - capsule_p0).sum(dim=-1)
        b = torch.mul(capsule_p1 - capsule_p0, capsule_q1 - capsule_q0).sum(dim=-1)
        c = torch.mul(capsule_q1 - capsule_q0, capsule_q1 - capsule_q0).sum(dim=-1)
        d = torch.mul(capsule_p1 - capsule_p0, capsule_p0 - capsule_q0).sum(dim=-1)
        e = torch.mul(capsule_q1 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        f = torch.mul(capsule_p0 - capsule_q0, capsule_p0 - capsule_q0).sum(dim=-1)
        # initialize s, t to zero
        s = torch.zeros(batch_size, num_edges//2, num_edges//2).to(capsule_p0.device)
        t = torch.zeros(batch_size, num_edges//2, num_edges//2).to(capsule_p0.device)
        one = torch.ones(batch_size, num_edges//2, num_edges//2).to(capsule_p0.device)
        # calculate coefficient
        det = a * c - b**2
        bte = b * e
        ctd = c * d
        ate = a * e
        btd = b * d
        # nonparallel segments
        # region 6
        s = torch.where((det > 0) & (bte <= ctd) & (e <= 0) & (-d >= a), one, s)
        s = torch.where((det > 0) & (bte <= ctd) & (e <= 0) & (-d < a) & (-d > 0), -d/a, s)
        # region 5
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e < c), e/c, t)
        # region 4
        s = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d >= a), one, s)
        s = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c) & (b - d < a) & (b - d > 0), (b-d)/a, s)
        t = torch.where((det > 0) & (bte <= ctd) & (e > 0) & (e >= c), one, t)
        # region 8
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e <= 0) & (-d > 0) & (-d < a), -d/a, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e <= 0) & (-d > 0) & (-d >= a), one, s)
        # region 1
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c), one, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e < c), (b+e)/c, t)
        # region 2
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c) & (b - d > 0) & (b - d < a), (b-d)/a, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c) & (b - d > 0) & (b - d >= a), one, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd >= det) & (b + e > 0) & (b + e >= c), one, t)
        # region 7
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate <= btd) & (-d > 0) & (-d >= a), one, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate <= btd) & (-d > 0) & (-d < a), -d/a, s)
        # region 3
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det) & (b - d > 0) & (b - d >= a), one, s)
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det) & (b - d > 0) & (b - d < a), (b-d)/a, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd >= det), one, t)
        # region 0
        s = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd < det), (bte-ctd)/det, s)
        t = torch.where((det > 0) & (bte > ctd) & (bte - ctd < det) & (ate > btd) & (ate - btd < det), (ate-btd)/det, t)
        # parallel segments
        # e <= 0
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d >= a), one, s)
        s = torch.where((det <= 0) & (e <= 0) & (-d > 0) & (-d < a), -d/a, s)
        # e >= c
        s = torch.where((det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d >= a), one, s)
        s = torch.where((det <= 0) & (e > 0) & (e >= c) & (b - d > 0) & (b - d < a), (b-d)/a, s)
        t = torch.where((det <= 0) & (e > 0) & (e >= c), one, t)
        # 0 < e < c
        t = torch.where((det <= 0) & (e > 0) & (e < c), e/c, t)
        # print(s, t)
        s = s.unsqueeze(-1).expand(batch_size, num_edges//2, num_edges//2, 3).detach()
        t = t.unsqueeze(-1).expand(batch_size, num_edges//2, num_edges//2, 3).detach()
        w = capsule_p0 - capsule_q0 + s*(capsule_p1 - capsule_p0) - t*(capsule_q1 - capsule_q0)
        dist_square = torch.mul(w, w).sum(dim=-1)
        return dist_square

"""
Joint Limit Loss
"""
class JointLimitLoss(nn.Module):
    def __init__(self):
        super(JointLimitLoss, self).__init__()
    
    def forward(self, ang, lower, upper):
        """
        Keyword auguments:
        ang -- joint angles [batch_size*num_nodes, num_node_features]
        """
        # calculate mask with limit
        lower_mask = ang < lower
        upper_mask = ang > upper
        
        # calculate final loss
        lower_loss = torch.sum(torch.masked_select(lower - ang, lower_mask))
        upper_loss = torch.sum(torch.masked_select(ang - upper, upper_mask))
        loss = (lower_loss + upper_loss)/ang.shape[0]
        
        return loss

"""
Regularization Loss
"""
class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, z):
        # calculate final loss
        batch_size = z.shape[0]
        loss = torch.mean(torch.norm(z.view(batch_size, -1), dim=1).pow(2))

        return loss


if __name__ == '__main__':
    fake_sphere = torch.Tensor([[[2,3,0]]])
    fake_capsule_p0 = torch.Tensor([[[0,0,0]]])
    fake_capsule_p1 = torch.Tensor([[[1,0,0]]])
    col_loss = CollisionLoss(threshold=1.0)
    # print(col_loss.sphere_capsule_dist_square(fake_sphere, fake_capsule_p0, fake_capsule_p1, 1, 2, 2))
    fake_capsule_p0 = torch.Tensor([[[0,0,0]]])
    fake_capsule_p1 = torch.Tensor([[[1,0,0]]])
    fake_capsule_q0 = torch.Tensor([[[-10,0,0]]])
    fake_capsule_q1 = torch.Tensor([[[-9,2,0]]])
    print(col_loss.capsule_capsule_dist_square(fake_capsule_p0, fake_capsule_p1, fake_capsule_q0, fake_capsule_q1, 1, 2))
