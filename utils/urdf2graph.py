import torch
import torch.nn as nn
from torch_geometric.data import Data
from urdfpy import URDF, matrix_to_xyz_rpy
import math


"""
convert Yumi URDF to graph
"""
def yumi2graph(urdf_file, cfg):
    # load URDF
    robot = URDF.load(urdf_file)

    # parse joint params
    joints = {}
    for joint in robot.joints:
        # joint atributes
        joints[joint.name] = {'type': joint.joint_type, 'axis': joint.axis,
                              'parent': joint.parent, 'child': joint.child,
                              'origin': matrix_to_xyz_rpy(joint.origin),
                              'lower': joint.limit.lower if joint.limit else 0,
                              'upper': joint.limit.upper if joint.limit else 0}

    # debug msg
    # for name, attr in joints.items():
    #     print(name, attr)

    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0

    # collect edge index & edge feature
    joints_name = cfg['joints_name']
    joints_index = {name: i for i, name in enumerate(joints_name)}
    edge_index = []
    edge_attr = []
    for edge in cfg['edges']:
        parent, child = edge
        # add edge index
        edge_index.append(torch.LongTensor([joints_index[parent], joints_index[child]]))
        # add edge attr
        edge_attr.append(torch.Tensor(joints[child]['origin']))
    edge_index = torch.stack(edge_index, dim=0)
    edge_index = edge_index.permute(1, 0)
    edge_attr = torch.stack(edge_attr, dim=0)
    # print(edge_index, edge_attr, edge_index.shape, edge_attr.shape)

    # number of nodes
    num_nodes = len(joints_name)

    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg['end_effectors']:
        ee_mask[joints_index[ee]] = True

    # shoulder mask
    sh_mask = torch.zeros(len(joints_name), 1).bool()
    for sh in cfg['shoulders']:
        sh_mask[joints_index[sh]] = True

    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg['elbows']:
        el_mask[joints_index[el]] = True

    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]

    # node offset
    offset = torch.stack([torch.Tensor(joints[joint]['origin']) for joint in joints_name], dim=0)
    # change root offset to store init pose
    init_pose = {}
    fk = robot.link_fk()
    for link, matrix in fk.items():
        init_pose[link.name] = matrix_to_xyz_rpy(matrix)
    origin = torch.zeros(6)
    for root in cfg['root_name']:
        offset[joints_index[root]] = torch.Tensor(init_pose[joints[root]['child']])
        origin[:3] += offset[joints_index[root]][:3]
    origin /= 2
    # move relative to origin
    for root in cfg['root_name']:
        offset[joints_index[root]] -= origin
    # print(offset, offset.shape)

    # dist to root
    root_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist
    # print(root_dist, root_dist.shape)

    # dist to shoulder
    shoulder_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and joints_name[current_idx] not in cfg['shoulders']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        shoulder_dist[node_idx] = dist
    # print(shoulder_dist, shoulder_dist.shape)

    # dist to elbow
    elbow_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while current_idx != -1 and joints_name[current_idx] not in cfg['elbows']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist
    # print(elbow_dist, elbow_dist.shape)

    # rotation axis
    axis = [torch.Tensor(joints[joint]['axis']) for joint in joints_name]
    axis = torch.stack(axis, dim=0)

    # joint limit
    lower = [torch.Tensor([joints[joint]['lower']]) for joint in joints_name]
    lower = torch.stack(lower, dim=0)
    upper = [torch.Tensor([joints[joint]['upper']]) for joint in joints_name]
    upper = torch.stack(upper, dim=0)
    # print(lower.shape, upper.shape)

    # skeleton
    data = Data(x=torch.zeros(num_nodes, 1),
                edge_index=edge_index,
                edge_attr=edge_attr,
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
                offset=offset,
                axis=axis,
                lower=lower,
                upper=upper)

    # test forward kinematics
    # print(joints_name)
    # result = robot.link_fk(cfg={joint:0.0 for joint in joints_name})
    # for link, matrix in result.items():
    #     print(link.name, matrix)
    # import os, sys, inspect
    # currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # parentdir = os.path.dirname(currentdir)
    # sys.path.insert(0,parentdir)
    # from models.kinematics import ForwardKinematicsURDF
    # fk = ForwardKinematicsURDF()
    # pos = fk(data.x, data.parent, data.offset, 1)

    # # visualize
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # # plot 3D lines
    # for edge in edge_index.permute(1, 0):
    #     line_x = [pos[edge[0]][0], pos[edge[1]][0]]
    #     line_y = [pos[edge[0]][1], pos[edge[1]][1]]
    #     line_z = [pos[edge[0]][2], pos[edge[1]][2]]
    #     plt.plot(line_x, line_y, line_z, 'royalblue', marker='o')
    # # plt.show()
    # plt.savefig('foo.png')

    return data

"""
convert Inspire Hand URDF graph
"""
def hand2graph(urdf_file, cfg):
    # load URDF
    robot = URDF.load(urdf_file)

    # parse joint params
    joints = {}
    for joint in robot.joints:
        # joint atributes
        joints[joint.name] = {'type': joint.joint_type, 'axis': joint.axis,
                              'parent': joint.parent, 'child': joint.child,
                              'origin': matrix_to_xyz_rpy(joint.origin),
                              'lower': joint.limit.lower if joint.limit else 0,
                              'upper': joint.limit.upper if joint.limit else 0}

    # debug msg
    # for name, attr in joints.items():
    #     print(name, attr)

    # skeleton type & topology type
    skeleton_type = 0
    topology_type = 0

    # collect edge index & edge feature
    joints_name = cfg['joints_name']
    joints_index = {name: i for i, name in enumerate(joints_name)}
    edge_index = []
    edge_attr = []
    for edge in cfg['edges']:
        parent, child = edge
        # add edge index
        edge_index.append(torch.LongTensor([joints_index[parent], joints_index[child]]))
        # add edge attr
        edge_attr.append(torch.Tensor(joints[child]['origin']))
    edge_index = torch.stack(edge_index, dim=0)
    edge_index = edge_index.permute(1, 0)
    edge_attr = torch.stack(edge_attr, dim=0)
    # print(edge_index, edge_attr, edge_index.shape, edge_attr.shape)

    # number of nodes
    num_nodes = len(joints_name)
    # print(num_nodes)

    # end effector mask
    ee_mask = torch.zeros(len(joints_name), 1).bool()
    for ee in cfg['end_effectors']:
        ee_mask[joints_index[ee]] = True
    # print(ee_mask)

    # elbow mask
    el_mask = torch.zeros(len(joints_name), 1).bool()
    for el in cfg['elbows']:
        el_mask[joints_index[el]] = True
    # print(el_mask)

    # node parent
    parent = -torch.ones(len(joints_name)).long()
    for edge in edge_index.permute(1, 0):
        parent[edge[1]] = edge[0]
    # print(parent)

    # node offset
    offset = []
    for joint in joints_name:
        offset.append(torch.Tensor(joints[joint]['origin']))
    offset = torch.stack(offset, dim=0)
    # print(offset, offset.shape)

    # dist to root
    root_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while joints_name[current_idx] != cfg['root_name']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        root_dist[node_idx] = dist
    # print(root_dist, root_dist.shape)

    # dist to elbow
    elbow_dist = torch.zeros(len(joints_name), 1)
    for node_idx in range(len(joints_name)):
        dist = 0
        current_idx = node_idx
        while joints_name[current_idx] != cfg['root_name'] and joints_name[current_idx] not in cfg['elbows']:
            origin = offset[current_idx]
            offsets_mod = math.sqrt(origin[0]**2+origin[1]**2+origin[2]**2)
            dist += offsets_mod
            current_idx = parent[current_idx]
        elbow_dist[node_idx] = dist
    # print(elbow_dist, elbow_dist.shape)

    # rotation axis
    axis = [torch.Tensor(joints[joint]['axis']) if joint != cfg['root_name'] else torch.zeros(3) for joint in joints_name]
    axis = torch.stack(axis, dim=0)
    # print(axis, axis.shape)

    # joint limit
    lower = [torch.Tensor([joints[joint]['lower']]) if joint != cfg['root_name'] else torch.zeros(1) for joint in joints_name]
    lower = torch.stack(lower, dim=0)
    upper = [torch.Tensor([joints[joint]['upper']]) if joint != cfg['root_name'] else torch.zeros(1) for joint in joints_name]
    upper = torch.stack(upper, dim=0)
    # print(lower, upper, lower.shape, upper.shape)

    # skeleton
    data = Data(x=torch.zeros(num_nodes, 1),
                edge_index=edge_index,
                edge_attr=edge_attr,
                skeleton_type=skeleton_type,
                topology_type=topology_type,
                ee_mask=ee_mask,
                el_mask=el_mask,
                root_dist=root_dist,
                elbow_dist=elbow_dist,
                num_nodes=num_nodes,
                parent=parent,
                offset=offset,
                axis=axis,
                lower=lower,
                upper=upper)
    # data for arm with hand
    data.hand_x = data.x
    data.hand_edge_index = data.edge_index
    data.hand_edge_attr = data.edge_attr
    data.hand_ee_mask = data.ee_mask
    data.hand_el_mask = data.el_mask
    data.hand_root_dist = data.root_dist
    data.hand_elbow_dist = data.elbow_dist
    data.hand_num_nodes = data.num_nodes
    data.hand_parent = data.parent
    data.hand_offset = data.offset
    data.hand_axis = data.axis
    data.hand_lower = data.lower
    data.hand_upper = data.upper
    # print(data)

    # # test forward kinematics
    # result = robot.link_fk(cfg={joint: 0.0 for joint in cfg['joints_name'] if joint != cfg['root_name']})
    # # for link, matrix in result.items():
    # #     print(link.name, matrix)
    # fk = ForwardKinematicsAxis()
    # pos, _, _ = fk(data.x, data.parent, data.offset, 1, data.axis)
    # # print(joints_index, pos)

    # # visualize
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.set_axis_off()
    # # ax.view_init(elev=0, azim=90)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim3d(-0.2,0.)
    # ax.set_ylim3d(-0.1,0.1)
    # ax.set_zlim3d(-0.1,0.1)

    # # plot 3D lines
    # for edge in edge_index.permute(1, 0):
    #     line_x = [pos[edge[0]][0], pos[edge[1]][0]]
    #     line_y = [pos[edge[0]][1], pos[edge[1]][1]]
    #     line_z = [pos[edge[0]][2], pos[edge[1]][2]]
    #     # line_x = [pos[edge[0]][2], pos[edge[1]][2]]
    #     # line_y = [pos[edge[0]][0], pos[edge[1]][0]]
    #     # line_z = [pos[edge[0]][1], pos[edge[1]][1]]
    #     plt.plot(line_x, line_y, line_z, 'royalblue', marker='o')
    # plt.show()
    # # plt.savefig('hand.png')

    return data

if __name__ == '__main__':
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
    graph = yumi2graph(urdf_file='./data/target/yumi/yumi.urdf', cfg=yumi_cfg)
    print('yumi', graph)

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
    graph = hand2graph(urdf_file='./data/target/yumi-with-hands/yumi_with_hands.urdf', cfg=hand_cfg)
    print('hand', graph)
