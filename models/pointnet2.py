import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, trunk_size, branch_sizes, output_size, n_layers, n_hidden, n_head):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, trunk_size + 2, [n_hidden//8, n_hidden//8, n_hidden//4], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, n_hidden//4 + 2, [n_hidden//4, n_hidden//4, n_hidden//2], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, n_hidden//2 + 2, [n_hidden//2, n_hidden//2, n_hidden], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, n_hidden + 2, [n_hidden, n_hidden, n_hidden*2], False)
        self.fp4 = PointNetFeaturePropagation(n_hidden*3, [n_hidden, n_hidden])
        self.fp3 = PointNetFeaturePropagation(3*n_hidden//2, [n_hidden, n_hidden])
        self.fp2 = PointNetFeaturePropagation(445, [n_hidden, n_hidden//2])
        self.fp1 = PointNetFeaturePropagation(n_hidden//2, [n_hidden//2, n_hidden//2, n_hidden//2])
        self.conv1 = nn.Conv1d(n_hidden//2, n_hidden//2, 1)
        self.bn1 = nn.BatchNorm1d(n_hidden//2)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(n_hidden//2, output_size, 1)

    def forward(self, xyz, inputs):
        
        xyz = xyz.permute(0,2,1)
        
        l0_points = xyz
        l0_xyz = xyz[:,:2,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x
    
    
    
def create_pointnet2(args):
    
    model = get_model(
        trunk_size=args.trunk_size,
        branch_sizes=args.branch_sizes,
        output_size=args.output_size,
        n_layers=args.n_layer,
        n_hidden=356,
        n_head=args.n_head,
    )
    
    return model