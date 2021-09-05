import torch
import torch.nn as nn

"""
Forward Kinematics for URDF
"""
class ForwardKinematicsURDF(nn.Module):
    def __init__(self):
        super(ForwardKinematicsURDF, self).__init__()

    def forward(self, x, parent, offset, num_graphs, axis='z', order='xyz'):
        """
        x -- joint angles [num_graphs*num_nodes, 1]
        parent -- node parent [num_graphs*num_nodes]
        offset -- node origin(xyzrpy) [num_graphs*num_nodes, 6]
        num_graphs -- number of graphs
        axis -- rotation axis for rotation x
        order -- rotation order for init rotation
        """
        x = x.view(num_graphs, -1) # [batch_size, num_nodes]
        parent = parent.view(num_graphs, -1)[0] # [num_nodes] the same batch, the same topology
        offset = offset.view(num_graphs, -1, 6) # [batch_size, num_nodes, 6]
        xyz = offset[:, :, :3] # [batch_size, num_nodes, 3]
        rpy = offset[:, :, 3:] # [batch_size, num_nodes, 3]

        positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        global_positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        rot_matrices = torch.empty(x.shape[0], x.shape[1], 3, 3, device=x.device) # [batch_size, num_nodes, 3, 3]
        transform = self.transform_from_axis(x, axis) # [batch_size, num_nodes, 3, 3]
        rpy_transform = self.transform_from_euler(rpy, order) # [batch_size, num_nodes, 3, 3]

        # iterate all nodes
        for node_idx in range(x.shape[1]):
            # serach parent
            parent_idx = parent[node_idx]

            # position
            if parent_idx != -1:
                positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + positions[:, parent_idx, :]
                global_positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + global_positions[:, parent_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rot_matrices[:, parent_idx, :, :].clone(), torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :]))
            else:
                positions[:, node_idx, :] = torch.zeros(3) # xyz[:, node_idx, :]
                global_positions[:, node_idx, :] = xyz[:, node_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :])

        return positions.view(-1, 3), rot_matrices.view(-1, 3, 3), global_positions.view(-1, 3)

    @staticmethod
    def transform_from_euler(rotation, order):
        transform = torch.matmul(ForwardKinematicsURDF.transform_from_axis(rotation[..., 2], order[2]),
                                 ForwardKinematicsURDF.transform_from_axis(rotation[..., 1], order[1]))
        transform = torch.matmul(transform,
                                 ForwardKinematicsURDF.transform_from_axis(rotation[..., 0], order[0]))
        return transform

    @staticmethod
    def transform_from_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform


"""
Forward Kinematics with Different Axes
"""
class ForwardKinematicsAxis(nn.Module):
    def __init__(self):
        super(ForwardKinematicsAxis, self).__init__()

    def forward(self, x, parent, offset, num_graphs, axis, order='xyz'):
        """
        x -- joint angles [num_graphs*num_nodes, 1]
        parent -- node parent [num_graphs*num_nodes]
        offset -- node origin(xyzrpy) [num_graphs*num_nodes, 6]
        num_graphs -- number of graphs
        axis -- rotation axis for rotation x
        order -- rotation order for init rotation
        """
        x = x.view(num_graphs, -1) # [batch_size, num_nodes]
        parent = parent.view(num_graphs, -1)[0] # [num_nodes] the same batch, the same topology
        axis = axis.view(num_graphs, -1, 3)[0] # [num_nodes, 3] the same batch, the same topology
        axis_norm = torch.norm(axis, dim=-1)
        # print(x.shape, axis.shape)
        x = x * axis_norm # filter no rotation node
        offset = offset.view(num_graphs, -1, 6) # [batch_size, num_nodes, 6]
        xyz = offset[:, :, :3] # [batch_size, num_nodes, 3]
        rpy = offset[:, :, 3:] # [batch_size, num_nodes, 3]

        positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        global_positions = torch.empty(x.shape[0], x.shape[1], 3, device=x.device) # [batch_size, num_nodes, 3]
        rot_matrices = torch.empty(x.shape[0], x.shape[1], 3, 3, device=x.device) # [batch_size, num_nodes, 3, 3]
        transform = self.transform_from_multiple_axis(x, axis) # [batch_size, num_nodes, 3, 3]
        rpy_transform = self.transform_from_euler(rpy, order) # [batch_size, num_nodes, 3, 3]

        # iterate all nodes
        for node_idx in range(x.shape[1]):
            # serach parent
            parent_idx = parent[node_idx]

            # position
            if parent_idx != -1:
                positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + positions[:, parent_idx, :]
                global_positions[:, node_idx, :] = torch.bmm(rot_matrices[:, parent_idx, :, :], xyz[:, node_idx, :].unsqueeze(2)).squeeze() + global_positions[:, parent_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rot_matrices[:, parent_idx, :, :].clone(), torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :]))
            else:
                positions[:, node_idx, :] = torch.zeros(3) # xyz[:, node_idx, :]
                global_positions[:, node_idx, :] = xyz[:, node_idx, :]
                rot_matrices[:, node_idx, :, :] = torch.bmm(rpy_transform[:, node_idx, :, :], transform[:, node_idx, :, :])

        return positions.view(-1, 3), rot_matrices.view(-1, 3, 3), global_positions.view(-1, 3)

    @staticmethod
    def transform_from_euler(rotation, order):
        transform = torch.matmul(ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 2], order[2]),
                                 ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 1], order[1]))
        transform = torch.matmul(transform,
                                 ForwardKinematicsAxis.transform_from_single_axis(rotation[..., 0], order[0]))
        return transform

    @staticmethod
    def transform_from_single_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        cord = ord(axis) - ord('x')

        transform[..., cord, :] = transform[..., :, cord] = 0
        transform[..., cord, cord] = 1

        if axis == 'x':
            transform[..., 1, 1] = transform[..., 2, 2] = cos
            transform[..., 1, 2] = -sin
            transform[..., 2, 1] = sin
        if axis == 'y':
            transform[..., 0, 0] = transform[..., 2, 2] = cos
            transform[..., 0, 2] = sin
            transform[..., 2, 0] = -sin
        if axis == 'z':
            transform[..., 0, 0] = transform[..., 1, 1] = cos
            transform[..., 0, 1] = -sin
            transform[..., 1, 0] = sin

        return transform

    @staticmethod
    def transform_from_multiple_axis(euler, axis):
        transform = torch.empty(euler.shape[0:2] + (3, 3), device=euler.device) # [batch_size, num_nodes, 3, 3]
        cos = torch.cos(euler)
        sin = torch.sin(euler)
        n1 = axis[..., 0]
        n2 = axis[..., 1]
        n3 = axis[..., 2]

        transform[..., 0, 0] = cos + n1 * n1 * (1 - cos)
        transform[..., 0, 1] = n1 * n2 * (1 - cos) - n3 * sin
        transform[..., 0, 2] = n1 * n3 * (1 - cos) + n2 * sin
        transform[..., 1, 0] = n1 * n2 * (1 - cos) + n3 * sin
        transform[..., 1, 1] = cos + n2 * n2 * (1 - cos)
        transform[..., 1, 2] = n2 * n3 * (1 - cos) - n1 * sin
        transform[..., 2, 0] = n1 * n3 * (1 - cos) - n2 * sin
        transform[..., 2, 1] = n2 * n3 * (1 - cos) + n1 * sin
        transform[..., 2, 2] = cos + n3 * n3 * (1 - cos)

        return transform