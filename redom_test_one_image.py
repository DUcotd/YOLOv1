import torch
import os
from PIL import Image
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from utils import cellboxes_to_boxes, non_max_suppression, plot_image
from dataset import VOCDataset
from model import Yolov1
from train import Compose

VOC_DIR = r"D:\StudyFile\DL\Pytorch_YoLo_From_Scratch\v1\scripts\data\voc\VOC_Detection"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
checkpoint = torch.load("overfit.pth.tar", weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
test_dataset = VOCDataset(
    voc_dir=VOC_DIR,
    mode="test",
    transform=transform
)

for image_num in range(4):
    idx = random.randint(0, len(test_dataset))


    image, label = test_dataset[idx]
    image_id = test_dataset.img_index[idx]
    image_path = os.path.abspath(os.path.join(VOC_DIR, "test", "images", image_id))
    original_image = Image.open(image_path)

    ret = model(image.unsqueeze(0).to(DEVICE))
    boxes = cellboxes_to_boxes(ret)

    bboxes = non_max_suppression(boxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
    plot_image(original_image, bboxes)

