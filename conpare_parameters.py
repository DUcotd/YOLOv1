import torch

from torchvision import models
from utils import count_parameters
from model import Yolov1



backbone = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
model = Yolov1(split_size=7, num_boxes=2, num_classes=20)
# count_parameters(model)
model_list = [layer for layer in model.children()]
model_no_linear = torch.nn.Sequential(*model_list[:-1])
count_parameters(model_no_linear, "model_no_linear")
count_parameters(backbone, "ShuffleNet_V2_X0_5")
