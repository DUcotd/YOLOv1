"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
import csv
from PIL import Image


index2label = ["person",
                "bird", "cat", "cow", "dog", "horse", "sheep",
                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, voc_dir, mode="train", S=7, B=2, C=20, transform=None,
    ):
        self.voc_dir = voc_dir
        self.mode = mode
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

        self.img_index = [img_id for img_id in os.listdir(os.path.join(self.voc_dir, f"{mode}", "images"))]
        self.label_dir = [label_id for label_id in os.listdir(os.path.join(self.voc_dir, f"{mode}", "targets"))]

        assert len(self.img_index) == len(self.label_dir)


    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, index):
        img_id = self.img_index[index]
        img_path = os.path.join(self.voc_dir, f"{self.mode}", "images", img_id)
        image = Image.open(img_path)
        
        
        img_path = os.path.join(self.voc_dir, f"{self.mode}", "images", img_id)
        label_path = os.path.join(self.voc_dir, f"{self.mode}", "targets", img_id[:-4] + ".csv")

        boxes = []
        with open(label_path) as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  
            for label in csv_reader:
                class_label, x, y, width, height = index2label.index(label[0]), *[float(label[i]) for i in range(1, 5)]
                x, y, width, height = self.convert(image.size, [x, y, width, height])
                boxes.append([class_label, x, y, width, height])

        
        boxes = torch.tensor(boxes)
        
        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
    
    def convert(self, size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[2])/2.0
        y = (box[1] + box[3])/2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        return (x,y,w,h)


if __name__ == "__main__":
    from torchvision.transforms import transforms
    class Compose(object):
        def __init__(self, transforms):
            self.transforms = transforms

        def __call__(self, img, bboxes):
            for t in self.transforms:
                img, bboxes = t(img), bboxes

            return img, bboxes


    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
    pascal_voc = VOCDataset(
        voc_dir=r"D:\StudyFile\DL\Pytorch_YoLo_From_Scratch\v1\scripts\data\voc\VOC_Detection",
        transform=transform
    )


    image, label = pascal_voc[0]
    print(image.shape)
