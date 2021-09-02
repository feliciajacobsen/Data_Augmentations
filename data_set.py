import numpy as np
import os
import pandas as pd 
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob


def make_labels():
    filenames = os.listdir("./data/images/")
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == "dog":
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        "filename": filenames,
        "category": categories,
    })


    labels_dir = "./data/labels/"
    if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

    df.to_csv(path_or_buf=labels_dir + "/labels.csv",index=False)


class CatsDogsDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir 
        self.transform = transform
        self.images = os.listdir(images_dir)

        if "dog" in self.images[0]:
            self.label = 1.0
        else:
            self.label = 0.0


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_paths = [f for f in glob.glob(self.images_dir+"*.jpg")]
        image = Image.open(image_paths[idx])

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label

my_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])



data_set = CatsDogsDataset("./data/images/", transform=my_transforms)


train_loader = DataLoader(
        data_set,
        batch_size = 10,
        shuffle = True,
    )




if __name__ == "__main__":
    pass
    #make_labels()
