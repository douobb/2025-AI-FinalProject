import torch
from torchvision import transforms
from skimage.transform import resize
from torch.utils.data import Dataset
from typing import List, Tuple
import os
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KDTree
import pandas as pd

# Define the color progression
colors = [
    "#b4b4b4",
    "#ffffff",
    "#e1ffe1", 
    "#2aff2a",  
    "#FFFF00",  
    "#ff7000",  
    "#ff0d00",
    "#990000",  
    "#9933cc",  
]

# Create custom colormap
seismic_custom = LinearSegmentedColormap.from_list("seismic_custom", colors, N=256)

# register to use
if "seismic_custom" not in colormaps:
    colormaps.register(name="seismic_custom", cmap=seismic_custom)

def image_to_scalar_matrix(image_path):
    # Load the RGB image as float32
    rgb_image = np.array(Image.open(image_path).convert('RGB')) / 255.0  # shape: (H, W, 3)

    h, w, _ = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3) # turn each pixel into a row of RGB values, ready for processing

    # Build a reference lookup: 256 colors in the gist_rainbow colormap (used in dataset)
    cmap = plt.get_cmap('seismic_custom')
    color_table = cmap(np.linspace(0, 1, 256))[:, :3]  # ignore alpha

    # Use KDTree's Nearest neighbour search to find nearest color match
    tree = KDTree(color_table)
    dist, indices = tree.query(pixels, k=1)

    # Convert index to value (0 to 1 range)
    scalar_values = indices.flatten() / 255.0
    scalar_matrix = scalar_values.reshape(h, w)

    return scalar_matrix  # shape (H, W), float32 in [0, 1]

def save_prediction_as_image(prediction_tensor, output_path: str):
    """
    Converts a predicted matrix back to an RGB image
    params:
        prediction_tensor: shape (1, H, W) or (H, W), float values in [0, 1]
        output_path: output image file path (should end in .jpg or .png)
    """
    # Detach and convert to NumPy if input is a tensor
    if isinstance(prediction_tensor, torch.Tensor):
        prediction_tensor = prediction_tensor.squeeze().detach().cpu().numpy()  # shape (H, W)

    # Ensure the matrix is in [0, 1] range
    prediction_tensor = np.clip(prediction_tensor, 0, 1)

    # Apply gist_rainbow colormap
    cmap = plt.get_cmap('seismic_custom')
    colored_img = cmap(prediction_tensor)[:, :, :3]  # Drop alpha channel, shape (H, W, 3)

    # Convert to 8-bit RGB
    rgb_img = (colored_img * 255).astype(np.uint8)

    # Save image using PIL
    im = Image.fromarray(rgb_img)
    im.save(output_path)

class MyDataset(Dataset):
    def __init__(self, target_images, latitudes, longitudes, depths, magnitudes):
        self.images = target_images
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.depths = depths
        self.magnitudes = magnitudes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        target_matrix = image_to_scalar_matrix(image_path)
        
        # tranforms
        target_matrix = resize(target_matrix, (60, 36), anti_aliasing=True)
        target_matrix = torch.tensor(target_matrix, dtype=torch.float32).unsqueeze(0)

        features = torch.tensor([
            self.latitudes[idx],
            self.longitudes[idx],
            self.depths[idx],
            self.magnitudes[idx]
        ], dtype=torch.float32)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        return target_matrix, features, base_name
    
def load_train_dataset(images_path: str='Final Project\Intensity_Generator\data_split\sea_removed\\train'):
    latitudes = []
    longitudes = []
    depths = []
    magnitudes = []

    # Convert each column to a list
    df = pd.read_csv('Final Project\Intensity_Generator\Earthquake.csv')
    label_lat = dict(zip(df['Image'], df['Lat']))
    label_lon = dict(zip(df['Image'], df['Lon']))
    label_dep = dict(zip(df['Image'], df['Depth']))
    label_mag = dict(zip(df['Image'], df['Mag']))

    target_images = []

    for filename in os.listdir(images_path):
        img_name = filename.rsplit('.', 1)[0]
        img_path = os.path.join(images_path, filename)
        target_images.append(img_path)
        # use dictionary to find corresponding attributes
        latitudes.append(label_lat[img_name])
        longitudes.append(label_lon[img_name])
        depths.append(label_dep[img_name])
        magnitudes.append(label_mag[img_name])

    return target_images, latitudes, longitudes, depths, magnitudes 

def load_test_dataset(images_path: str='Final Project\Intensity_Generator\data_split\sea_removed\\test'):
    latitudes = []
    longitudes = []
    depths = []
    magnitudes = []

    # Convert each column to a list
    df = pd.read_csv('Final Project\Intensity_Generator\Earthquake.csv')
    label_lat = dict(zip(df['Image'], df['Lat']))
    label_lon = dict(zip(df['Image'], df['Lon']))
    label_dep = dict(zip(df['Image'], df['Depth']))
    label_mag = dict(zip(df['Image'], df['Mag']))

    target_images = []

    for filename in os.listdir(images_path):
        img_name = filename.rsplit('.', 1)[0]
        img_path = os.path.join(images_path, filename)
        target_images.append(img_path)
        # use dictionary to find corresponding attributes
        latitudes.append(label_lat[img_name])
        longitudes.append(label_lon[img_name])
        depths.append(label_dep[img_name])
        magnitudes.append(label_mag[img_name])

    return target_images, latitudes, longitudes, depths, magnitudes


def plot(train_losses: List, val_losses: List):
    # (TODO) Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    #        xlabel: 'Epoch', ylabel: 'Loss'
    plt.title('Training and Validation Loss')
    plt.plot(train_losses, label='Train Loss', marker='x')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()

    print("Save the plot to 'loss.png'")
    return