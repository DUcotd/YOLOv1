import torch
from torchvision import models

from model import Yolov1
from utils import count_parameters

def get_model():
    backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    backbone = [layer for layer in backbone.children()][:-1]
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
    model = [layer for layer in model.children()][-1]
    model[1] = torch.nn.Linear(200704, 496)
    backbone.append(model)
    model = torch.nn.Sequential(*backbone)
    return model

# backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
# backbone = [layer for layer in backbone.children()][:-1]
# backbone.append(torch.nn.Flatten())
# model = torch.nn.Sequential(*backbone)

model = get_model()
x = torch.randn(1, 3, 448, 448)
print(model(x).shape)
count_parameters(model, "model")