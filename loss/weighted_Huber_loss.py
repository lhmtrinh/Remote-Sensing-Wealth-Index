import torch
import numpy as np
import torch.nn.functional as F

class WeightedHuberLoss(torch.nn.modules.loss._Loss):
    def __init__(self, dense_weight_model, delta=1.0, reduction='mean'):
        super(WeightedHuberLoss, self).__init__(reduction=reduction)
        self.dense_weight_model = dense_weight_model
        self.delta = delta
        
    def forward(self, input, target):

        weights = torch.tensor(self.dense_weight_model.dense_weight(target.cpu().numpy()), dtype=torch.float32, device=input.device)
        
        # Calculate the Huber loss for each sample, without reduction
        huber_loss = F.smooth_l1_loss(input, target, reduction='none', beta=self.delta)
        
        # Apply the sample weights to the Huber loss
        weighted_huber_loss = weights * huber_loss
        
        # Apply the specified reduction method to the weighted loss
        if self.reduction == 'mean':
            return torch.mean(weighted_huber_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_huber_loss)
        else:
            return weighted_huber_loss