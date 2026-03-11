import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from model.ov_ai_model import OV_MLP, OV_RESNET, OV_TRANSFORMER


trained_model_file = "obj_verf_tanh_ep_3000_lr_0_001_4_layer_batch_norm_equal_fold_0_best_model.pth"

model = OV_RESNET(input_size = 660,hidden_layers= [512, 512, 256, 256, 128, 128])
checkpoint = torch.load(trained_model_file)
model.load_state_dict(checkpoint)
for param in model.state_dict():
    print(f"{0}: {param}")

weights_layer1 = model.state_dict()["input_layer.weight"]
bias_layer1 = model.state_dict()["input_layer.bias"]

print(weights_layer1.shape)

sns.heatmap(weights_layer1.numpy(), cmap="coolwarm", annot=False)
plt.title("Heatmap derGewichte (Layer1)")
#plt.show()

feature_importance = np.sum(np.abs(weights_layer1.numpy()), axis = 0)
print(feature_importance)
plt.bar(range(len(feature_importance)), feature_importance)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importance basierend auf Gewichten")
plt.show()

def detect_outliers_iqr(data):
    Q1 = np.percentile(data,25)
    Q3 = np.percentile(data, 75)

    IQR = Q3-Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("lower bound: ", lower_bound)

    return np.where((data< lower_bound))

outliers = detect_outliers_iqr(feature_importance)[0]

print(outliers.shape)