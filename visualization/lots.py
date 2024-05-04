import torch
from utils.utils import normalize
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def LOTS(imageinit, iterations, model, get_feature_maps, device,tau=0.1, alpha=0.1):
    # 'imageinit' is the initial image tensor
    # 'Ft' is the target feature vector (often a zero tensor for visualization)
    # 'iter' is the number of iterations
    # 'tau' is the distance threshold
    # 'alpha' is the step size scaling factor
    imageadv = imageinit.clone().requires_grad_(True)
    for _ in range(iterations):
        model.zero_grad()
        Fs = get_feature_maps(imageadv.to(device))
        Ft = torch.zeros(Fs.shape).to(device)
        distance = torch.norm(Fs - Ft)
        if distance > tau:
            loss = F.mse_loss(Fs, Ft)
            loss.backward()
            gradient = imageadv.grad.data
            gradient_step = alpha * gradient / gradient.abs().max()
            imageadv.data -= gradient_step
            imageadv.data.clamp_(0, 1)  # Assuming image pixel values are in the range [0, 1]
            imageadv.grad.data.zero_()
        else:
            break
    return imageadv.detach(), distance  # Detach the image from the current graph to prevent further gradient computation

def calculate_activation_map(imageinit, imageadv, filter_size, normalize = False):
    # imageinit size [C, H, W]
    perturbation = torch.abs(imageinit - imageadv).mean(dim = 0) # mean across channels
    if not normalize:
        return perturbation
    perturbation_norm = normalize(perturbation)
    perturbation_blurred = TF.gaussian_blur(perturbation_norm.unsqueeze(0).unsqueeze(0),
                                        kernel_size=[filter_size, filter_size],
                                        sigma=(1.5, 1.5)).squeeze()
    # Normalize again after blurring
    activation_map = normalize(perturbation_blurred)
    return activation_map