import torch
import torch.nn.functional as F

class WeightedL3Loss(torch.nn.modules.loss._Loss):
    def __init__(self, dense_weight_model, size_average=None, reduce=None, reduction='mean'):
        super(WeightedL3Loss, self).__init__(size_average, reduce, reduction)
        self.dense_weight_model = dense_weight_model

    def forward(self, input, target):
        weights = torch.tensor(self.dense_weight_model.dense_weight(target.cpu().numpy()), dtype=torch.float32, device=input.device)
        mse_loss = F.mse_loss(input, target, reduction='none')

        with torch.no_grad():
            mae_loss = F.l1_loss(input, target, reduction= 'none')

        weighted_l3_loss = weights * mse_loss * mae_loss
        
        if self.reduction == 'mean':
            return torch.mean(weighted_l3_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_l3_loss)
        else:
            return weighted_l3_loss
        

class L3Loss(torch.nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(L3Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        mse_loss = F.mse_loss(input, target, reduction='none')

        with torch.no_grad():
            mae_loss = F.l1_loss(input, target, reduction= 'none')

        l3_loss = mse_loss * mae_loss
        
        if self.reduction == 'mean':
            return torch.mean(l3_loss)
        elif self.reduction == 'sum':
            return torch.sum(l3_loss)
        else:
            return l3_loss