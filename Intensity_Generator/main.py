import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils import shuffle
from loguru import logger

from CNN import IntensityNet, train, validate, test
from utils import MyDataset, load_train_dataset, load_test_dataset, plot

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

    EPOCHS = 10
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
    model.save("/content/drive/MyDrive/AI_2025/AI_Final_Project/IntenistyNetFor Earthquake_model.h5")
    logger.info(f"model saved to google drive.")

    """
    CNN - test
    """
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_loss, test_acc = test(model, test_loader, criterion, device)

    # Print the training log 
    logger.info(f"Val Loss:   {test_loss:.4f}")
    logger.info(f"Val Accuracy:    {test_acc:.4f}")

if __name__ == '__main__':
    main()
