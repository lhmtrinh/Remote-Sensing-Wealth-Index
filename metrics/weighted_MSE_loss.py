import torch
import torch.nn.functional as F

class WeightedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, dense_weight_model, size_average=None, reduce=None, reduction='mean'):
        super(WeightedMSELoss, self).__init__(size_average, reduce, reduction)
        self.dense_weight_model = dense_weight_model

    def forward(self, input, target):
        weights = torch.tensor(self.dense_weight_model.dense_weight(target.cpu().numpy()), dtype=torch.float32, device=input.device)
        mse_loss = F.mse_loss(input, target, reduction='none')
        weighted_mse_loss = weights * mse_loss
        if self.reduction == 'mean':
            return torch.mean(weighted_mse_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_mse_loss)
        else:
            return weighted_mse_loss
