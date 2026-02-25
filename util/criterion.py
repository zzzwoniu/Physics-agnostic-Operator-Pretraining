import torch
from torch.nn.modules.loss import _WeightedLoss

import math as mt


class SimpleLpLoss(_WeightedLoss):
    def __init__(self, d=2, p=2, size_average=True, reduction=True,return_comps = False):
        super(SimpleLpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.return_comps = return_comps



    def forward(self, x, y, weight=None, mask=None):
        num_examples = x.size()[0]

        # if weight is not None:
        #     x = x * weight
        #     y = y * weight
        
        # Lp loss 1
        if mask is not None:##TODO: will be meaned by n_channels for single channel data
            x = x * mask
            y = y * mask

            ## compute effective channels
            # msk_channels = mask.sum(dim=(1,2,3),keepdim=False).count_nonzero(dim=-1) # B, 1
            msk_channels = mask.sum(dim=list(range(1, mask.ndim-1)),keepdim=False).count_nonzero(dim=-1) # B, 1
        else:
            msk_channels = x.shape[-1]

        diff_norms = torch.norm(x.reshape(num_examples,-1, x.shape[-1]) - y.reshape(num_examples,-1,x.shape[-1]), self.p, dim=1)    ##N, C
        y_norms = torch.norm(y.reshape(num_examples,-1, y.shape[-1]), self.p, dim=1) + 1e-8

        if self.reduction:
            if self.size_average:
                if weight is None:
                    return torch.mean(diff_norms/y_norms)          
                else:
                    return torch.sum(torch.mean(weight*diff_norms/y_norms, dim=-1))/(torch.sum(weight) + 1e-4)
            else:
                return diff_norms/y_norms
                # return torch.sum(torch.sum(diff_norms/y_norms, dim=-1) / msk_channels)    #### go this branch
        else:
            return torch.sum(diff_norms/y_norms, dim=-1) / msk_channels
        
        
class MSELoss(_WeightedLoss):
    def __init__(self, size_average=True):
        super(MSELoss, self).__init__()
        
        self.size_average = size_average
        
    def forward(self, x, y, mask=None):
        
        if mask is not None:
            x = x * mask
            y = y * mask
            
        diff = x - y
            
        return torch.mean(diff**2)