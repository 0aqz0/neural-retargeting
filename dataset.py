import torch
import torch_geometric.transforms as transforms
from torch_geometric.data import Data as OldData
from torch_geometric.data import InMemoryDataset

import os
import math
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R
from utils.urdf2graph import yumi2graph, hand2graph
import h5py


class Data(OldData):
    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.num_nodes
        elif key == 'l_hand_edge_index':
            return self.l_hand_num_nodes
        elif key == 'r_hand_edge_index':
            return self.r_hand_num_nodes
        else:
            return 0

"""
Normalize by a constant coefficient
"""
class Normalize(object):
    def __call__(self, data, coeff=100.0):
        if hasattr(data, 'x'):
            data.x = data.x/coeff
        if hasattr(data, 'l_hand_x'):
            data.l_hand_x = data.l_hand_x/coeff
        if hasattr(data, 'r_hand_x'):
            data.r_hand_x = data.r_hand_x/coeff
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

"""
Target Dataset for Yumi Manipulator
"""
class YumiDataset(InMemoryDataset):
    yumi_cfg = {
        'joints_name': [
            'yumi_joint_1_l',
            'yumi_joint_2_l',
            'yumi_joint_7_l',
            'yumi_joint_3_l',
            'yumi_joint_4_l',
            'yumi_joint_5_l',
            'yumi_joint_6_l',
            'yumi_joint_1_r',
            'yumi_joint_2_r',
            'yumi_joint_7_r',
            'yumi_joint_3_r',
            'yumi_joint_4_r',
            'yumi_joint_5_r',
            'yumi_joint_6_r',
        ],
        'edges': [
            ['yumi_joint_1_l', 'yumi_joint_2_l'],
            ['yumi_joint_2_l', 'yumi_joint_7_l'],
            ['yumi_joint_7_l', 'yumi_joint_3_l'],
            ['yumi_joint_3_l', 'yumi_joint_4_l'],
            ['yumi_joint_4_l', 'yumi_joint_5_l'],
            ['yumi_joint_5_l', 'yumi_joint_6_l'],
            ['yumi_joint_1_r', 'yumi_joint_2_r'],
            ['yumi_joint_2_r', 'yumi_joint_7_r'],
            ['yumi_joint_7_r', 'yumi_joint_3_r'],
            ['yumi_joint_3_r', 'yumi_joint_4_r'],
            ['yumi_joint_4_r', 'yumi_joint_5_r'],
            ['yumi_joint_5_r', 'yumi_joint_6_r'],
        ],
        'root_name': [
            'yumi_joint_1_l',
            'yumi_joint_1_r',
        ],
        'end_effectors': [
            'yumi_joint_6_l',
            'yumi_joint_6_r',
        ],
        'shoulders': [
            'yumi_joint_2_l',
            'yumi_joint_2_r',
        ],
        'elbows': [
            'yumi_joint_3_l',
            'yumi_joint_3_r',
        ],
    }
    def __init__(self, root, transform=None, pre_transform=None):
        super(YumiDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self._raw_file_names = [os.path.join(self.root, file) for file in os.listdir(self.root) if file.endswith('.urdf')]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data_list.append(yumi2graph(file, self.yumi_cfg))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
Map glove data to inspire hand data
"""
def linear_map(x_, min_, max_, min_hat, max_hat):
    x_hat = 1.0 * (x_ - min_) / (max_ - min_) * (max_hat - min_hat) + min_hat
    return x_hat

def map_glove_to_inspire_hand(glove_angles):

    ### This function linearly maps the Wiseglove angle measurement to Inspire hand's joint angles.

    ## preparation, specify the range for linear scaling
    hand_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1, 0.0, 0.0]) # radius already
    hand_final = np.array([-1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -1.6, -0.75, 0.0, -0.2, -0.15])
    glove_start = np.array([0, 0, 53, 0, 0, 22, 0, 0, 22, 0, 0, 35, 0, 0])# * pi / 180.0 # degree to radius
    glove_final = np.array([45, 100, 0, 90, 120, 0, 90, 120, 0, 90, 120, 0, 90, 120])# * pi / 180.0
    length = glove_angles.shape[0]
    hand_angles = np.zeros((length, 12)) # 12 joints

    ## Iterate to map angles
    for i in range(length):
        # four fingers' extension/flexion (abduction/adduction are dumped)
        hand_angles[i, 0] = linear_map(glove_angles[i, 3], glove_start[3], glove_final[3], hand_start[0], hand_final[0]) # Link1 (joint name)
        hand_angles[i, 1] = linear_map(glove_angles[i, 4], glove_start[4], glove_final[4], hand_start[1], hand_final[1]) # Link11
        hand_angles[i, 2] = linear_map(glove_angles[i, 6], glove_start[6], glove_final[6], hand_start[2], hand_final[2]) # Link2
        hand_angles[i, 3] = linear_map(glove_angles[i, 7], glove_start[7], glove_final[7], hand_start[3], hand_final[3]) # Link22
        hand_angles[i, 4] = linear_map(glove_angles[i, 9], glove_start[9], glove_final[9], hand_start[4], hand_final[4]) # Link3
        hand_angles[i, 5] = linear_map(glove_angles[i, 10], glove_start[10], glove_final[10], hand_start[5], hand_final[5]) # Link33
        hand_angles[i, 6] = linear_map(glove_angles[i, 12], glove_start[12], glove_final[12], hand_start[6], hand_final[6]) # Link4
        hand_angles[i, 7] = linear_map(glove_angles[i, 13], glove_start[13], glove_final[13], hand_start[7], hand_final[7]) # Link44

        # thumb
        hand_angles[i, 8] = (hand_start[8] + hand_final[8]) / 2.0 # Link5 (rotation about z axis), fixed!
        hand_angles[i, 9] = linear_map(glove_angles[i, 2], glove_start[2], glove_final[2], hand_start[9], hand_final[9]) # Link 51
        hand_angles[i, 10] = linear_map(glove_angles[i, 0], glove_start[0], glove_final[0], hand_start[10], hand_final[10]) # Link 52
        hand_angles[i, 11] = linear_map(glove_angles[i, 1], glove_start[1], glove_final[1], hand_start[11], hand_final[11]) # Link 53

    return hand_angles

"""
Parse H5 File
"""
def parse_h5(filename, selected_key=None):
    data_list = []
    h5_file = h5py.File(filename, 'r')
    # print(filename, h5_file.keys(), len(h5_file.keys()))
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key and selected_key is None:
            print('Skipping'+key)
            continue
        # glove data
        l_glove_angle = h5_file[key + '/l_glove_angle'][:]
        r_glove_angle = h5_file[key + '/r_glove_angle'][:]
        l_hand_angle = map_glove_to_inspire_hand(l_glove_angle)
        r_hand_angle = map_glove_to_inspire_hand(r_glove_angle)
        # position data
        l_shoulder_pos = h5_file[key + '/l_up_pos'][:]
        r_shoulder_pos = h5_file[key + '/r_up_pos'][:]
        l_elbow_pos = h5_file[key + '/l_fr_pos'][:]
        r_elbow_pos = h5_file[key + '/r_fr_pos'][:]
        l_wrist_pos = h5_file[key + '/l_hd_pos'][:]
        r_wrist_pos = h5_file[key + '/r_hd_pos'][:]
        # quaternion data
        l_shoulder_quat = R.from_quat(h5_file[key + '/l_up_quat'][:])
        r_shoulder_quat = R.from_quat(h5_file[key + '/r_up_quat'][:])
        l_elbow_quat = R.from_quat(h5_file[key + '/l_fr_quat'][:])
        r_elbow_quat = R.from_quat(h5_file[key + '/r_fr_quat'][:])
        l_wrist_quat = R.from_quat(h5_file[key + '/l_hd_quat'][:])
        r_wrist_quat = R.from_quat(h5_file[key + '/r_hd_quat'][:])
        # rotation matrix data
        l_shoulder_matrix = l_shoulder_quat.as_matrix()
        r_shoulder_matrix = r_shoulder_quat.as_matrix()
        l_elbow_matrix = l_elbow_quat.as_matrix()
        r_elbow_matrix = r_elbow_quat.as_matrix()
        l_wrist_matrix = l_wrist_quat.as_matrix()
        r_wrist_matrix = r_wrist_quat.as_matrix()
        # transform to local coordinates
        # l_wrist_matrix = l_wrist_matrix * inv(l_elbow_matrix)
        # r_wrist_matrix = r_wrist_matrix * inv(r_elbow_matrix)
        # l_elbow_matrix = l_elbow_matrix * inv(l_shoulder_matrix)
        # r_elbow_matrix = r_elbow_matrix * inv(r_shoulder_matrix)
        # l_shoulder_matrix = l_shoulder_matrix * inv(l_shoulder_matrix)
        # r_shoulder_matrix = r_shoulder_matrix * inv(r_shoulder_matrix)
        # euler data
        l_shoulder_euler = R.from_matrix(l_shoulder_matrix).as_euler('zyx', degrees=True)
        r_shoulder_euler = R.from_matrix(r_shoulder_matrix).as_euler('zyx', degrees=True)
        l_elbow_euler = R.from_matrix(l_elbow_matrix).as_euler('zyx', degrees=True)
        r_elbow_euler = R.from_matrix(r_elbow_matrix).as_euler('zyx', degrees=True)
        l_wrist_euler = R.from_matrix(l_wrist_matrix).as_euler('zyx', degrees=True)
        r_wrist_euler = R.from_matrix(r_wrist_matrix).as_euler('zyx', degrees=True)

        total_frames = l_shoulder_pos.shape[0]
        for t in range(total_frames):
            data = parse_arm(l_shoulder_euler[t], l_elbow_euler[t], l_wrist_euler[t], r_shoulder_euler[t], r_elbow_euler[t], r_wrist_euler[t],
                            l_shoulder_pos[t], l_elbow_pos[t], l_wrist_pos[t], r_shoulder_pos[t], r_elbow_pos[t], r_wrist_pos[t],
                            l_shoulder_quat[t], l_elbow_quat[t], l_wrist_quat[t], r_shoulder_quat[t], r_elbow_quat[t], r_wrist_quat[t])
            data_list.append(data)
    return data_list, l_hand_angle, r_hand_angle

def parse_arm(l_shoulder_euler, l_elbow_euler, l_wrist_euler, r_shoulder_euler, r_elbow_euler, r_wrist_euler,
            l_shoulder_pos, l_elbow_pos, l_wrist_pos, r_shoulder_pos, r_elbow_pos, r_wrist_pos,
            l_shoulder_quat, l_elbow_quat, l_wrist_quat, r_shoulder_quat, r_elbow_quat, r_wrist_quat):
    # x
    x = torch.stack([torch.from_numpy(l_shoulder_euler),
                        torch.from_numpy(l_elbow_euler),
                        torch.from_numpy(l_wrist_euler),
                        torch.from_numpy(r_shoulder_euler),
                        torch.from_numpy(r_elbow_euler),
                        torch.from_numpy(r_wrist_euler)], dim=0).float()
    # number of nodes
    num_nodes = 6
    # edge index
    edge_index = torch.LongTensor([[0, 1, 3, 4],
                                    [1, 2, 4, 5]])
    # position
    pos = torch.stack([torch.from_numpy(l_shoulder_pos),
                        torch.from_numpy(l_elbow_pos),
                        torch.from_numpy(l_wrist_pos),
                        torch.from_numpy(r_shoulder_pos),
                        torch.from_numpy(r_elbow_pos),
                        torch.from_numpy(r_wrist_pos)], dim=0).float()
    # edge attributes
    edge_attr = []
    for edge in edge_index.permute(1, 0):
        parent = edge[0]
        child = edge[1]
        edge_attr.append(pos[child] - pos[parent])
    edge_attr = torch.stack(edge_attr, dim=0)
    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0
    # end effector mask
    ee_mask = torch.zeros(num_nodes, 1).bool()
    ee_mask[2] = ee_mask[5] = True
    # shoulder mask
    sh_mask = torch.zeros(num_nodes, 1).bool()
    sh_mask[0] = sh_mask[3] = True
    # elbow mask
    el_mask = torch.zeros(num_nodes, 1).bool()
    el_mask[1] = el_mask[4] = True
    # parent
    parent = torch.LongTensor([-1, 0, 1, -1, 3, 4])
    # offset
    offset = torch.zeros(num_nodes, 3)
    for node_idx in range(num_nodes):
        if parent[node_idx] != -1:
            offset[node_idx] = pos[node_idx] - pos[parent[node_idx]]
        else:
            offset[node_idx] = pos[node_idx]
    # distance to root
    root_dist = torch.zeros(num_nodes, 1)
    for node_idx in range(num_nodes):
        dist = 0
        current_idx = node_idx
        while current_idx != -1:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist
    # distance to shoulder
    shoulder_dist = torch.zeros(num_nodes, 1)
    for node_idx in range(num_nodes):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and current_idx != 0 and current_idx != 3:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        shoulder_dist[node_idx] = dist
    # distance to elbow
    elbow_dist = torch.zeros(num_nodes, 1)
    for node_idx in range(num_nodes):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and current_idx != 1 and current_idx != 4:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist
    # quaternion
    q = torch.stack([torch.from_numpy(l_shoulder_quat.as_quat()),
                        torch.from_numpy(l_elbow_quat.as_quat()),
                        torch.from_numpy(l_wrist_quat.as_quat()),
                        torch.from_numpy(r_shoulder_quat.as_quat()),
                        torch.from_numpy(r_elbow_quat.as_quat()),
                        torch.from_numpy(r_wrist_quat.as_quat())], dim=0).float()
    data = Data(x=torch.cat([x,pos], dim=-1),
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                q=q,
                skeleton_type=skeleton_type,
                topology_type=topology_type,
                ee_mask=ee_mask,
                sh_mask=sh_mask,
                el_mask=el_mask,
                root_dist=root_dist,
                shoulder_dist=shoulder_dist,
                elbow_dist=elbow_dist,
                num_nodes=num_nodes,
                parent=parent,
                offset=offset)
    # print(data)
    return data

"""
Source Dataset for Sign Language
"""
class SignDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SignDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_path = os.path.join(self.root, 'h5')
        self._raw_file_names = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data, _, _ = parse_h5(file)
            data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
parse h5 with hand
"""
def parse_h5_hand(filename, selected_key=None):
    data_list = []
    h5_file = h5py.File(filename, 'r')
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key and selected_key is None:
            print('Skipping'+key)
            continue
        # glove data
        l_glove_pos = h5_file[key + '/l_glove_pos'][:]
        r_glove_pos = h5_file[key + '/r_glove_pos'][:]
        # insert zero for root
        total_frames = l_glove_pos.shape[0]
        l_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), l_glove_pos], axis=1)
        r_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), r_glove_pos], axis=1)
        # print(l_glove_pos.shape, r_glove_pos.shape)
        # switch dimensions
        l_glove_pos = np.stack([-l_glove_pos[..., 2], -l_glove_pos[..., 1], -l_glove_pos[..., 0]], axis=-1)
        r_glove_pos = np.stack([-r_glove_pos[..., 2], -r_glove_pos[..., 1], -r_glove_pos[..., 0]], axis=-1)

        for t in range(total_frames):
            data = parse_glove_pos(l_glove_pos[t])
            data.l_hand_x = data.x
            data.l_hand_edge_index = data.edge_index
            data.l_hand_edge_attr = data.edge_attr
            data.l_hand_pos = data.pos
            data.l_hand_ee_mask = data.ee_mask
            data.l_hand_el_mask = data.el_mask
            data.l_hand_root_dist = data.root_dist
            data.l_hand_elbow_dist = data.elbow_dist
            data.l_hand_num_nodes = data.num_nodes
            data.l_hand_parent = data.parent
            data.l_hand_offset = data.offset

            r_hand_data = parse_glove_pos(r_glove_pos[t])
            data.r_hand_x = r_hand_data.x
            data.r_hand_edge_index = r_hand_data.edge_index
            data.r_hand_edge_attr = r_hand_data.edge_attr
            data.r_hand_pos = r_hand_data.pos
            data.r_hand_ee_mask = r_hand_data.ee_mask
            data.r_hand_el_mask = r_hand_data.el_mask
            data.r_hand_root_dist = r_hand_data.root_dist
            data.r_hand_elbow_dist = r_hand_data.elbow_dist
            data.r_hand_num_nodes = r_hand_data.num_nodes
            data.r_hand_parent = r_hand_data.parent
            data.r_hand_offset = r_hand_data.offset

            data_list.append(data)
    return data_list

def parse_glove_pos(glove_pos):
    # x
    x = torch.from_numpy(glove_pos).float()

    # number of nodes
    num_nodes = 17

    # edge index
    edge_index = torch.LongTensor([[0, 1, 2, 0, 4, 5, 0, 7, 8,  0, 10, 11,  0, 13, 14, 15],
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])

    # position
    pos = torch.from_numpy(glove_pos).float()

    # edge attributes
    edge_attr = []
    for edge in edge_index.permute(1, 0):
        parent = edge[0]
        child = edge[1]
        edge_attr.append(pos[child] - pos[parent])
    edge_attr = torch.stack(edge_attr, dim=0)

    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0

    # end effector mask
    ee_mask = torch.zeros(num_nodes, 1).bool()
    ee_mask[3] = ee_mask[6] = ee_mask[9] = ee_mask[12] = ee_mask[16] = True

    # elbow mask
    el_mask = torch.zeros(num_nodes, 1).bool()
    el_mask[1] = el_mask[4] = el_mask[7] = el_mask[10] = el_mask[13] = True

    # parent
    parent = torch.LongTensor([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15])

    # offset
    offset = torch.zeros(num_nodes, 3)
    for node_idx in range(num_nodes):
        if parent[node_idx] != -1:
            offset[node_idx] = pos[node_idx] - pos[parent[node_idx]]

    # distance to root
    root_dist = torch.zeros(num_nodes, 1)
    for node_idx in range(num_nodes):
        dist = 0
        current_idx = node_idx
        while parent[current_idx] != -1:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist

    # distance to elbow
    elbow_dist = torch.zeros(num_nodes, 1)
    for node_idx in range(num_nodes):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and not el_mask[current_idx]:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist

    data = Data(x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                skeleton_type=skeleton_type,
                topology_type=topology_type,
                ee_mask=ee_mask,
                el_mask=el_mask,
                root_dist=root_dist,
                elbow_dist=elbow_dist,
                num_nodes=num_nodes,
                parent=parent,
                offset=offset)
    # print(data)
    return data


"""
Source Dataset for Sign Language with Hand
"""
class SignWithHand(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SignWithHand, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_path = os.path.join(self.root, 'h5')
        self._raw_file_names = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data = parse_h5_hand(file)
            data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
Target Dataset for Inspire Hand
"""
class InspireHand(InMemoryDataset):
    hand_cfg = {
        'joints_name': [
            'yumi_link_7_r_joint',
            'Link1',
            'Link11',
            'Link1111',
            'Link2',
            'Link22',
            'Link2222',
            'Link3',
            'Link33',
            'Link3333',
            'Link4',
            'Link44',
            'Link4444',
            'Link5',
            'Link51',
            'Link52',
            'Link53',
            'Link5555',
        ],
        'edges': [
            ['yumi_link_7_r_joint', 'Link1'],
            ['Link1', 'Link11'],
            ['Link11', 'Link1111'],
            ['yumi_link_7_r_joint', 'Link2'],
            ['Link2', 'Link22'],
            ['Link22', 'Link2222'],
            ['yumi_link_7_r_joint', 'Link3'],
            ['Link3', 'Link33'],
            ['Link33', 'Link3333'],
            ['yumi_link_7_r_joint', 'Link4'],
            ['Link4', 'Link44'],
            ['Link44', 'Link4444'],
            ['yumi_link_7_r_joint', 'Link5'],
            ['Link5', 'Link51'],
            ['Link51', 'Link52'],
            ['Link52', 'Link53'],
            ['Link53', 'Link5555'],
        ],
        'root_name': 'yumi_link_7_r_joint',
        'end_effectors': [
            'Link1111',
            'Link2222',
            'Link3333',
            'Link4444',
            'Link5555',
        ],
        'elbows': [
            'Link1',
            'Link2',
            'Link3',
            'Link4',
            'Link5',
        ],
    }
    def __init__(self, root, transform=None, pre_transform=None):
        super(InspireHand, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self._raw_file_names = [os.path.join(self.root, file) for file in os.listdir(self.root) if file.endswith('.urdf')]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data_list.append(hand2graph(file, self.hand_cfg))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
parse h5 with all data
"""
def parse_all(filename, selected_key=None):
    data_list = []
    h5_file = h5py.File(filename, 'r')
    if selected_key is None:
        keys = h5_file.keys()
    else:
        keys = [selected_key]
    for key in keys:
        if '语句' in key and selected_key is None:
            print('Skipping'+key)
            continue
        # position data
        l_shoulder_pos = h5_file[key + '/l_up_pos'][:]
        r_shoulder_pos = h5_file[key + '/r_up_pos'][:]
        l_elbow_pos = h5_file[key + '/l_fr_pos'][:]
        r_elbow_pos = h5_file[key + '/r_fr_pos'][:]
        l_wrist_pos = h5_file[key + '/l_hd_pos'][:]
        r_wrist_pos = h5_file[key + '/r_hd_pos'][:]
        # quaternion data
        l_shoulder_quat = R.from_quat(h5_file[key + '/l_up_quat'][:])
        r_shoulder_quat = R.from_quat(h5_file[key + '/r_up_quat'][:])
        l_elbow_quat = R.from_quat(h5_file[key + '/l_fr_quat'][:])
        r_elbow_quat = R.from_quat(h5_file[key + '/r_fr_quat'][:])
        l_wrist_quat = R.from_quat(h5_file[key + '/l_hd_quat'][:])
        r_wrist_quat = R.from_quat(h5_file[key + '/r_hd_quat'][:])
        # rotation matrix data
        l_shoulder_matrix = l_shoulder_quat.as_matrix()
        r_shoulder_matrix = r_shoulder_quat.as_matrix()
        l_elbow_matrix = l_elbow_quat.as_matrix()
        r_elbow_matrix = r_elbow_quat.as_matrix()
        l_wrist_matrix = l_wrist_quat.as_matrix()
        r_wrist_matrix = r_wrist_quat.as_matrix()
        # transform to local coordinates
        # l_wrist_matrix = l_wrist_matrix * inv(l_elbow_matrix)
        # r_wrist_matrix = r_wrist_matrix * inv(r_elbow_matrix)
        # l_elbow_matrix = l_elbow_matrix * inv(l_shoulder_matrix)
        # r_elbow_matrix = r_elbow_matrix * inv(r_shoulder_matrix)
        # l_shoulder_matrix = l_shoulder_matrix * inv(l_shoulder_matrix)
        # r_shoulder_matrix = r_shoulder_matrix * inv(r_shoulder_matrix)
        # euler data
        l_shoulder_euler = R.from_matrix(l_shoulder_matrix).as_euler('zyx', degrees=True)
        r_shoulder_euler = R.from_matrix(r_shoulder_matrix).as_euler('zyx', degrees=True)
        l_elbow_euler = R.from_matrix(l_elbow_matrix).as_euler('zyx', degrees=True)
        r_elbow_euler = R.from_matrix(r_elbow_matrix).as_euler('zyx', degrees=True)
        l_wrist_euler = R.from_matrix(l_wrist_matrix).as_euler('zyx', degrees=True)
        r_wrist_euler = R.from_matrix(r_wrist_matrix).as_euler('zyx', degrees=True)
        # glove data
        l_glove_pos = h5_file[key + '/l_glove_pos'][:]
        r_glove_pos = h5_file[key + '/r_glove_pos'][:]
        # insert zero for root
        total_frames = l_glove_pos.shape[0]
        l_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), l_glove_pos], axis=1)
        r_glove_pos = np.concatenate([np.zeros((total_frames, 1, 3)), r_glove_pos], axis=1)
        # print(l_glove_pos.shape, r_glove_pos.shape)
        # switch dimensions
        l_glove_pos = np.stack([-l_glove_pos[..., 2], -l_glove_pos[..., 1], -l_glove_pos[..., 0]], axis=-1)
        r_glove_pos = np.stack([-r_glove_pos[..., 2], -r_glove_pos[..., 1], -r_glove_pos[..., 0]], axis=-1)

        for t in range(total_frames):
            data = Data()
            l_hand_data = parse_glove_pos(l_glove_pos[t])
            data.l_hand_x = l_hand_data.x
            data.l_hand_edge_index = l_hand_data.edge_index
            data.l_hand_edge_attr = l_hand_data.edge_attr
            data.l_hand_pos = l_hand_data.pos
            data.l_hand_ee_mask = l_hand_data.ee_mask
            data.l_hand_el_mask = l_hand_data.el_mask
            data.l_hand_root_dist = l_hand_data.root_dist
            data.l_hand_elbow_dist = l_hand_data.elbow_dist
            data.l_hand_num_nodes = l_hand_data.num_nodes
            data.l_hand_parent = l_hand_data.parent
            data.l_hand_offset = l_hand_data.offset

            r_hand_data = parse_glove_pos(r_glove_pos[t])
            data.r_hand_x = r_hand_data.x
            data.r_hand_edge_index = r_hand_data.edge_index
            data.r_hand_edge_attr = r_hand_data.edge_attr
            data.r_hand_pos = r_hand_data.pos
            data.r_hand_ee_mask = r_hand_data.ee_mask
            data.r_hand_el_mask = r_hand_data.el_mask
            data.r_hand_root_dist = r_hand_data.root_dist
            data.r_hand_elbow_dist = r_hand_data.elbow_dist
            data.r_hand_num_nodes = r_hand_data.num_nodes
            data.r_hand_parent = r_hand_data.parent
            data.r_hand_offset = r_hand_data.offset

            arm_data = parse_arm(l_shoulder_euler[t], l_elbow_euler[t], l_wrist_euler[t], r_shoulder_euler[t], r_elbow_euler[t], r_wrist_euler[t],
                            l_shoulder_pos[t], l_elbow_pos[t], l_wrist_pos[t], r_shoulder_pos[t], r_elbow_pos[t], r_wrist_pos[t],
                            l_shoulder_quat[t], l_elbow_quat[t], l_wrist_quat[t], r_shoulder_quat[t], r_elbow_quat[t], r_wrist_quat[t])
            data.x = arm_data.x
            data.edge_index = arm_data.edge_index
            data.edge_attr = arm_data.edge_attr
            data.pos = arm_data.pos
            data.q = arm_data.q
            data.skeleton_type = arm_data.skeleton_type
            data.topology_type = arm_data.topology_type
            data.ee_mask = arm_data.ee_mask
            data.sh_mask = arm_data.sh_mask
            data.el_mask = arm_data.el_mask
            data.root_dist = arm_data.root_dist
            data.shoulder_dist = arm_data.shoulder_dist
            data.elbow_dist = arm_data.elbow_dist
            data.num_nodes = arm_data.num_nodes
            data.parent = arm_data.parent
            data.offset = arm_data.offset
            data_list.append(data)
    return data_list

"""
Source Dataset for Sign Language with Hand
"""
class SignAll(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SignAll, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        data_path = os.path.join(self.root, 'h5')
        self._raw_file_names = [os.path.join(data_path, file) for file in os.listdir(data_path)]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data = parse_all(file)
            data_list.extend(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
Target Dataset for Yumi
"""
class YumiAll(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YumiAll, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        self._raw_file_names = [os.path.join(self.root, file) for file in os.listdir(self.root) if file.endswith('.urdf')]
        return self._raw_file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for file in self.raw_file_names:
            data = yumi2graph(file, YumiDataset.yumi_cfg)
            hand_data = hand2graph(file, InspireHand.hand_cfg)
            data.hand_x = hand_data.x
            data.hand_edge_index = hand_data.edge_index
            data.hand_edge_attr = hand_data.edge_attr
            data.hand_ee_mask = hand_data.ee_mask
            data.hand_el_mask = hand_data.el_mask
            data.hand_root_dist = hand_data.root_dist
            data.hand_elbow_dist = hand_data.elbow_dist
            data.hand_num_nodes = hand_data.num_nodes
            data.hand_parent = hand_data.parent
            data.hand_offset = hand_data.offset
            data.hand_axis = hand_data.axis
            data.hand_lower = hand_data.lower
            data.hand_upper = hand_data.upper
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    yumi_dataset = YumiDataset(root='./data/target/yumi')
    sign_dataset = SignDataset(root='./data/source/sign/train', pre_transform=transforms.Compose([Normalize()]))
    inspire_hand = InspireHand(root='./data/target/yumi-with-hands')
    sign_with_hand = SignWithHand(root='./data/source/sign-hand/train', pre_transform=transforms.Compose([Normalize()]))
    sign_all = SignAll(root='./data/source/sign-all/train', pre_transform=transforms.Compose([Normalize()]))
    yumi_all = YumiAll(root='./data/target/yumi-all')
