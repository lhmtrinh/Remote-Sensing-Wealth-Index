regression with resnet 24 channels, RGB channels are scaled to imagenet
loss function is modified to tackle imbalance structure
from model import load_resnet_model
model = load_resnet_model('resnet50', num_classes=1)
from train_regression_weighted_loss import train_model
train_model(model, train_files, val_files, device, epochs=15)
