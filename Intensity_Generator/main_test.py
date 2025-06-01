import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.utils import shuffle
from sklearn.neighbors import KDTree
from loguru import logger
from tqdm import tqdm
from typing import List, Tuple
from skimage.transform import resize
import os
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

class IntensityNet(nn.Module):
    def __init__(self):
        super(IntensityNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 60 * 36),
        )

    def forward(self, x):
        x = self.fc(x)     # [batch_size, 60 * 36]
        x = x.view(-1, 1, 60, 36)  # [batch_size, channels=1, H=60, W=36]
        return x

def train(model: IntensityNet, train_loader: DataLoader, criterion, optimizer, device, epoch)->float: 
    save_root = 'data_split/prediction/train'
    model.train()
    total_loss = 0.0
    loop = tqdm(enumerate(train_loader), total=len(train_loader)) # batch_size = 32

    for i, (target_matrix, features, base_name) in loop:
        # move data to device (where model is located)
        target_matrix = target_matrix.to(device)
        features = features.to(device)

        # forward
        outputs = model(features) # [batch_size, channels=1, H=60, W=36]
        loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
        total_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save prediction
        if epoch == 4:
            for j in range(outputs.size(0)):
                filename = base_name[j]
                filename = f"{filename}.jpg"
                save_path = os.path.join(save_root, filename)
                save_prediction_as_image(outputs[j], save_path)

        # tqdm display
        loop.set_description(f"Training")
        loop.set_postfix(loss=loss.item()) # display loss (should decrease through training)

    avg_loss = total_loss / len(train_loader)

    return avg_loss

def validate(model: IntensityNet, val_loader: DataLoader, criterion, device, epoch)->Tuple[float, float]:
    save_root = 'data_split/prediction/train/'
    model.eval()
    total_loss = 0.0
    total_size = 0
    correct_size = 0 # percentage of "good enough" predictions

    loop = tqdm(enumerate(val_loader), total=len(val_loader)) # batch_size = 32
    with torch.no_grad():
        for i, (target_matrix, features, base_name) in loop:
            # move data to device (where model is located)
            target_matrix = target_matrix.to(device)
            features = features.to(device)

            # forward
            outputs = model(features) # [batch_size, channels=1, H=60, W=36]
            loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
            total_loss += loss.item()

            # validation
            total_size += outputs.size(0) # collect batch size
            for j in range(outputs.size(0)):
                if epoch == 4:
                    filename = base_name[j]
                    filename = f"{filename}.jpg"
                    save_path = os.path.join(save_root, filename)
                    save_prediction_as_image(outputs[j], save_path)

                if criterion(target_matrix[j], outputs[j]).item() < 0.05:
                    correct_size += 1

            # tqdm display
            loop.set_description(f"Validating")
            loop.set_postfix(loss=loss.item()) # display loss

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_size / total_size
    return avg_loss, accuracy

def test(model: IntensityNet, test_loader: DataLoader, criterion, device):
    save_root = 'data_split/prediction/test/'
    model.eval()

    total_loss = 0.0
    total_size = 0
    correct_size = 0 # percentage of "good enough" predictions

    loop = tqdm(enumerate(test_loader), total=len(test_loader)) # batch_size = 32
    with torch.no_grad():
        for i, (target_matrix, features, base_name) in loop:
            # move data to device (where model is located)
            target_matrix = target_matrix.to(device)
            features = features.to(device)

            # forward
            outputs = model(features) # [batch_size, channels=1, H=60, W=36]
            loss = criterion(outputs, target_matrix) # criterion = nn.SmoothL1Loss()
            total_size += outputs.size(0) # collect batch size
            total_loss += loss.item()

            # save predictions
            for j in range(outputs.size(0)):
                filename = base_name[j]
                filename = f"{filename}.jpg"
                save_path = os.path.join(save_root, filename)
                save_prediction_as_image(outputs[j], save_path)
                
                if criterion(target_matrix[j], outputs[j]).item() < 0.05:
                    correct_size += 1

            # tqdm display
            loop.set_description(f"Testing")
            loop.set_postfix(loss=loss.item()) # display loss

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_size / total_size
    return avg_loss, accuracy


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

# Optional: register to use with plt.get_cmap
if "seismic_custom" not in colormaps:
    colormaps.register(name="seismic_custom", cmap=seismic_custom)


def gist_rainbow_image_to_scalar_matrix(image_path):
    # Load the RGB image as float32
    rgb_image = np.array(Image.open(image_path).convert('RGB')) / 255.0  # shape: (H, W, 3)

    h, w, _ = rgb_image.shape
    pixels = rgb_image.reshape(-1, 3) # turn each pixel into a row of RGB values, ready for processing

    # Build a reference lookup: 256 colors in the gist_rainbow colormap (used in dataset)
    cmap = plt.get_cmap('seismic_custom', 256)
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
    Converts a predicted matrix (float tensor or numpy array) back to an RGB image using the gist_rainbow colormap.
    Saves the result as a JPG image.

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
        target_matrix = gist_rainbow_image_to_scalar_matrix(image_path)
        
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
    
def load_train_dataset(images_path: str='data_split/sea_removed/train/'):
    latitudes = []
    longitudes = []
    depths = []
    magnitudes = []

    # Convert each column to a list
    df = pd.read_csv('Earthquake.csv')
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

def load_test_dataset(images_path: str='data_split/sea_removed/test/'):
    latitudes = []
    longitudes = []
    depths = []
    magnitudes = []

    # Convert each column to a list
    df = pd.read_csv('Earthquake.csv')
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
    # Plot the training loss and validation loss of CNN, and save the plot to 'loss.png'
    # xlabel: 'Epoch', ylabel: 'Loss'
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

def main():
    """
    load data
    """
    logger.info("Start loading data")
    target_images, latitudes, longitudes, depths, magnitudes = load_train_dataset()
    target_images, latitudes, longitudes, depths, magnitudes = shuffle(target_images, latitudes, longitudes, depths, magnitudes, random_state=777)
    train_len = int(0.9 * len(target_images)) # 0.1 portion for validation

    train_images, val_images = target_images[:train_len], target_images[train_len:]
    train_latitudes, val_latitudes = latitudes[:train_len], latitudes[train_len:]
    train_longitudes, val_longitudes = longitudes[:train_len], longitudes[train_len:]
    train_depths, val_depths = depths[:train_len], depths[train_len:]
    train_magnitudes, val_magnitudes = magnitudes[:train_len], magnitudes[train_len:]
    
    test_images, test_latitudes, test_longitudes, test_depths, test_magnitudes = load_test_dataset()

    train_dataset = MyDataset(train_images, train_latitudes, train_longitudes, train_depths, train_magnitudes)
    val_dataset = MyDataset(val_images, val_latitudes, val_longitudes, val_depths, val_magnitudes)
    test_dataset = MyDataset(test_images, test_latitudes, test_longitudes, test_depths, test_magnitudes)
    
    """
    CNN - train and validate
    """
    logger.info("Start training CNN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = IntensityNet().to(device)
    criterion = nn.SmoothL1Loss(reduction='mean')

    # Optimizer configuration
    base_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(base_params, lr=1e-4)
    reduce_lr = ReduceLROnPlateau(optimizer=optimizer, factor=0.5,
                           patience=3, mode='min', verbose=1,
                           min_lr=1e-5)
    
    train_losses = []
    val_losses = []
    max_acc = 0.0

    EPOCHS = 5
    for epoch in range(EPOCHS): #epoch
        train_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        max_acc = max(max_acc, val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        reduce_lr.step(val_loss)

        # Print the training log 
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}]")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f}")
        logger.info(f"Val Accuracy:    {val_acc:.4f}")
        logger.info(f"Max Accuracy:   {max_acc:.4f}")

    logger.info(f"Best Accuracy: {max_acc:.4f}")

    """
    CNN - plot
    """
    plot(train_losses, val_losses)
    torch.save(model.state_dict(), "/content/drive/MyDrive/AI_2025/AI_Final_Project/IntensityNet_model.pth")
    logger.info(f"model weights saved to google drive.")

    """
    CNN - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    # Print the training log 
    logger.info(f"Test Loss:   {test_loss:.4f}")
    logger.info(f"Test Accuracy:    {test_acc:.4f}")

if __name__ == '__main__':
    main()
