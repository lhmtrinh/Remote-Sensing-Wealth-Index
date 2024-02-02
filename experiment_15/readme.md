like experiment 9 but with the new data using half precision
from model import load_resnet_model
from train_regression_weighted_loss import train_model
from weighted_MSE_loss import DenseWeight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_resnet_model('resnet50', num_classes=1)

dense_weight_model = DenseWeight(0.2)

Must do experiment 16 using the old data to compare too

Epoch 1/30, Train Loss: 0.2751, Train R2: -0.3411, Val Loss: 0.1900, Val R2: 0.1043
Model saved as checkpoint_epoch_1.pth
Epoch 2/30, Train Loss: 0.1272, Train R2: 0.3945, Val Loss: 0.1902, Val R2: 0.1104
Epoch 3/30, Train Loss: 0.1189, Train R2: 0.4335, Val Loss: 0.1788, Val R2: 0.1584
Model saved as checkpoint_epoch_3.pth
Epoch 4/30, Train Loss: 0.1107, Train R2: 0.4717, Val Loss: 0.2009, Val R2: 0.0504
Epoch 5/30, Train Loss: 0.1065, Train R2: 0.4910, Val Loss: 0.2446, Val R2: -0.1491
Epoch 6/30, Train Loss: 0.1000, Train R2: 0.5210, Val Loss: 0.2192, Val R2: -0.0263
Epoch 7/30, Train Loss: 0.0939, Train R2: 0.5501, Val Loss: 0.2337, Val R2: -0.0849
Epoch 8/30, Train Loss: 0.0884, Train R2: 0.5753, Val Loss: 0.1984, Val R2: 0.0810
Epoch 9/30, Train Loss: 0.0852, Train R2: 0.5901, Val Loss: 0.1437, Val R2: 0.3293
Model saved as checkpoint_epoch_9.pth
Epoch 10/30, Train Loss: 0.0778, Train R2: 0.6233, Val Loss: 0.2012, Val R2: 0.0563
Epoch 11/30, Train Loss: 0.0742, Train R2: 0.6409, Val Loss: 0.2059, Val R2: 0.0302
Epoch 12/30, Train Loss: 0.0715, Train R2: 0.6528, Val Loss: 0.2156, Val R2: -0.0118
Epoch 13/30, Train Loss: 0.0633, Train R2: 0.6913, Val Loss: 0.1983, Val R2: 0.0634
Epoch 14/30, Train Loss: 0.0597, Train R2: 0.7075, Val Loss: 0.1521, Val R2: 0.2725
Epoch 15/30, Train Loss: 0.0538, Train R2: 0.7368, Val Loss: 0.1497, Val R2: 0.2733
Epoch 16/30, Train Loss: 0.0539, Train R2: 0.7355, Val Loss: 0.1574, Val R2: 0.2264
Epoch 17/30, Train Loss: 0.0566, Train R2: 0.7217, Val Loss: 0.1432, Val R2: 0.3029
Model saved as checkpoint_epoch_17.pth
Epoch 18/30, Train Loss: 0.0588, Train R2: 0.7107, Val Loss: 0.1344, Val R2: 0.3560
Model saved as checkpoint_epoch_18.pth
Epoch 19/30, Train Loss: 0.0460, Train R2: 0.7726, Val Loss: 0.1251, Val R2: 0.4061
Model saved as checkpoint_epoch_19.pth
Epoch 20/30, Train Loss: 0.0487, Train R2: 0.7587, Val Loss: 0.1429, Val R2: 0.3249
Epoch 21/30, Train Loss: 0.0455, Train R2: 0.7744, Val Loss: 0.2227, Val R2: -0.0506
Epoch 22/30, Train Loss: 0.0500, Train R2: 0.7525, Val Loss: 0.1387, Val R2: 0.3435
Epoch 23/30, Train Loss: 0.0508, Train R2: 0.7471, Val Loss: 0.1410, Val R2: 0.3301
Epoch 24/30, Train Loss: 0.0406, Train R2: 0.7983, Val Loss: 0.1342, Val R2: 0.3675
Epoch 25/30, Train Loss: 0.0412, Train R2: 0.7956, Val Loss: 0.1366, Val R2: 0.3524
Epoch 26/30, Train Loss: 0.0456, Train R2: 0.7731, Val Loss: 0.1870, Val R2: 0.1282
Epoch 27/30, Train Loss: 0.0633, Train R2: 0.6855, Val Loss: 0.1388, Val R2: 0.3241
Epoch 28/30, Train Loss: 0.0525, Train R2: 0.7404, Val Loss: 0.1715, Val R2: 0.1579
Epoch 29/30, Train Loss: 0.0295, Train R2: 0.8545, Val Loss: 0.1434, Val R2: 0.3000
Epoch 30/30, Train Loss: 0.0162, Train R2: 0.9203, Val Loss: 0.1280, Val R2: 0.3968
Model saved as final_model.pth
Training completed and final model saved.


MAE for bin 1: 1.3468268
MAE for bin 2: 0.8179501
MAE for bin 3: 0.48292407
MAE for bin 4: 0.25444755
MAE for bin 5: 0.2045858
MAE for bin 6: 0.20702669
Weighted MAE: 1.1864262111399033