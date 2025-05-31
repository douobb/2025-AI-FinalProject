import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL
import pandas as pd

class TrainDataset(Dataset):
    def __init__(self, images, labels, extra_params):
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((300, 180)),
            transforms.ToTensor()
        ])
        self.images, self.labels, self.extra_params = images, labels, extra_params

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(self.labels[idx], dtype=torch.float32), torch.tensor(self.extra_params[idx], dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, image, extra_params):
        self.transform = transforms.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.Resize((300, 180)),
            transforms.ToTensor()
        ])
        self.image, self.extra_params = image, extra_params

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image_path = self.image[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return image, base_name, torch.tensor(self.extra_params[idx], dtype=torch.float32)
    
def load_train_dataset(target):
    if target == 'lon' or target == 'lat':
        path = 'data_split/processed_images_no_epicenter/train/'
    elif target == 'mag' or target == 'dep':
        path = 'data_split/processed_images/train/'
    else:
        print('error - wrong target name')
        return
    images = []
    labels = []
    extra_params = []
    
    df = pd.read_csv('Earthquake.csv')
    label_mag = dict(zip(df['Image'], df['Mag']))
    label_lon = dict(zip(df['Image'], df['Lon']))
    label_lat = dict(zip(df['Image'], df['Lat']))
    label_dep = dict(zip(df['Image'], df['Depth']))

    for filename in os.listdir(path):
        img_name = filename.rsplit('.', 1)[0]
        img_path = os.path.join(path, filename)
        images.append(img_path)
        if target == 'mag':
            labels.append(label_mag[img_name])
            extra_params.append([label_lon[img_name], label_lat[img_name], label_dep[img_name]])
        elif target == 'dep':
            labels.append(label_dep[img_name])
            extra_params.append([label_lon[img_name], label_lat[img_name], label_mag[img_name]])
        elif target == 'lon':
            labels.append(label_lon[img_name])
            extra_params.append([label_mag[img_name], label_dep[img_name]])
        else:
            labels.append(label_lat[img_name])
            extra_params.append([label_mag[img_name], label_dep[img_name]])
    return images, labels, extra_params

def load_test_dataset(target):
    if target == 'lon' or target == 'lat':
        path = 'data_split/processed_images_no_epicenter/test/'
    elif target == 'mag' or target == 'dep':
        path = 'data_split/processed_images/test/'
    else:
        print('error - wrong target name')
        return
    images = []
    extra_params = []

    df = pd.read_csv('Earthquake.csv')
    label_mag = dict(zip(df['Image'], df['Mag']))
    label_lon = dict(zip(df['Image'], df['Lon']))
    label_lat = dict(zip(df['Image'], df['Lat']))
    label_dep = dict(zip(df['Image'], df['Depth']))

    for filename in os.listdir(path):
        img_name = filename.rsplit('.', 1)[0]
        img_path = os.path.join(path, filename)
        images.append(img_path)
        if target == 'mag':
            extra_params.append([label_lon[img_name], label_lat[img_name], label_dep[img_name]])
        elif target == 'dep':
            extra_params.append([label_lon[img_name], label_lat[img_name], label_mag[img_name]])
        elif target == 'lon':
            extra_params.append([label_mag[img_name], label_dep[img_name]])
        else:
            extra_params.append([label_mag[img_name], label_dep[img_name]])
    return images, extra_params
