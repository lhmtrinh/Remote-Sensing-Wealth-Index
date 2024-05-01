import torch
import torch.nn.functional as F

class BMCLoss(torch.nn.modules.loss._Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)

def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
    pred: A float tensor of size [batch, 1].
    target: A float tensor of size [batch, 1].
    noise_var: A float number or tensor.
    Returns:
    loss: A float tensor. Balanced MSE Loss.
    """
    pred = pred.unsqueeze(dim=1)
    target = target.unsqueeze(dim=1)
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]

    device = pred.device
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=device))  # contrastive-like loss   
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss
