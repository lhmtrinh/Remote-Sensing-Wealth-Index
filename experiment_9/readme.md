regression with resnet 24 channels, RGB channels are NOT scaled to imagenet
loss function is modified to tackle imbalance structure

from model import load_resnet_model
from train_regression_weighted_loss import train_model
from weighted_MSE_loss import DenseWeight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_resnet_model('resnet50', num_classes=1)

dense_weight_model = DenseWeight(0.2)


