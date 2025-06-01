# Applications of Machine Learning in Earthquake Prediction
## Overview
### Goal
The objective of this project is to explore the potential of machine learning techniques in analyzing and predicting key earthquake parameters based on observed or estimated data. Earthquake prediction is a highly complex and nonlinear problem, where traditional physics-based models often fall short in flexibility and adaptability. Machine learning offers a data-driven alternative capable of capturing hidden patterns and complex relationships among seismic variables.

### Subtasks
* Predict magnitude based on intensity map, epicenter location and depth.
* Predict depth based on intensity map, epicenter location and magnitude.
* Predict epicenter location based on intensity map.
* Predict intensity distribution of an earthquake event bsed on magnitude, distance, site condition, and style-of-faulting (斷層分類).

### Data sources
[地震活動彙整 - 中央氣象署地震測報中心](https://scweb.cwa.gov.tw/zh-tw/earthquake/data)

## Prerequisite
* Python 3.13.3
* Packages: in [requirements.txt](https://github.com/douobb/2025-AI-FinalProject/blob/main/requirements.txt)

## Usage
### Download & Preprocess
Execute the following files in order:
1. `image_download.py` - Download images from the web  
2. `preprocess.py` - Preprocess and crop the images  
3. `data folder.py` - Split images into train/test folders  

After execution, the following folder structure should be created:
```
data/
├── raw_images/
│   ├── 2000/
│   │   ├── 2000001.png
│   │   ├── 2000002.png
│   │   └── ...
│   ├── ...
│   └── 2025/
├── processed_images/
│   ├── 2000/
│   ├── ...
│   └── 2025/
├── processed_images_no_epicenter/
│   ├── 2000/
│   ├── ...
│   └── 2025/
```

```
data_split/
├── processed_images/
│   ├── train/
│   │   ├── 2000001.png
│   │   ├── 2012015.png
│   │   └── ...
│   └── test/
│       ├── 2004012.png
│       ├── 2023007.png
│       └── ...
├── processed_images_no_epicenter/
│   ├── train/
│   └── test/
```
### For Predict (Regression)
1. Before starting, make sure that the `data_split` folder, `Earthquake.csv`, and the following Python files are in the same directory: `utils.py`, `CNN.py`, and `main.py`.
2. In `main.py`, you can configure the following hyperparameters:
   | Parameter | Possible Value | Description |
   | --- | --- | --- |
   | `target` | `'dep'`, `'mag'`, `'loc'`, `'lat'`, `'lon'` | The prediction target (e.g., depth, magnitude, or location) |
   | `nRound` | Positive integer | Number of repeated training rounds for averaging results |
   | `EPOCHS` | Positive integer | Number of training epochs per round |
   | `lr` | Small float | Learning rate used by the optimizer |
   | `model_type` | `'resnet18'`, `'efficientnet_b0'`, `'simple_cnn'` | The model architecture used for processing the input images |
   | `use_extra_params`| `True`, `False` | Whether to include additional input features (e.g., magnitude, depth)|
   | `criterion` | `nn.L1Loss()`, `nn.SmoothL1Loss()`, `nn.MSELoss()` | Loss function used during training |
   
4. Run `main.py`. For each round, a `.csv` file will be generated that records both the predicted and actual values on the test set.
5. After all rounds are completed, the script will print the average MAE across all rounds.


### For ...
TBD


