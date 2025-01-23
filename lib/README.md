# `lib` Folder

## Overview
The `lib` folder contains a set of scripts designed to process, fine-tune, and analyze face images, specifically using the CelebA dataset. This library provides utilities for downloading data, preprocessing images, generating embeddings, building queries, and uploading datasets. Additionally, it includes tools for visualization and handling machine learning pipelines.

## Folder Structure
```
lib/
├── constants.py       # Common constants.
├── prepare.py         # Data downloading utilities.
├── processing.py      # Face detection and preprocessing.
├── pipeline.py        # Main data processing pipeline.
├── embedding.py       # Embedding generation and query building.
├── upload_dataset.py  # Dataset uploading to Kaggle.
├── visualize.py       # Visualization utilities.
```

### Features
- **Data Preparation:** Download and preprocess the CelebA dataset, including face cropping.
- **Embedding Generation:** Generate feature embeddings using pre-trained models like MobileNetV2.
- **Query and Retrieval:** Build KD-trees for efficient image retrieval and execute query evaluations.
- **Visualization:** Plot images with bounding boxes for analysis.
- **Dataset Management:** Upload datasets to Kaggle.

## File Descriptions

### 1. `constants.py`
Defines reusable constants used across the library:
- `IMAGE_SHAPE`: The original shape of the CelebA images.
- `IMAGENET_IMAGE_SIZE`: Standard size used for resizing images for models.

### 2. `prepare.py`
Handles downloading the CelebA dataset using the Kaggle API. Returns the local path where the dataset is stored.

**Functionality:** `download_data`: Downloads the dataset from Kaggle.

### 3. `processing.py`
Contains utilities for preprocessing images, including face detection and cropping.

**Key Functions:**
- `crop_face`: Detects and crops faces from images using Haar cascades.
- `face_coordination`: Extracts and resizes face regions.


### 4. `pipeline.py`
Defines the main data processing pipeline.

**Functionality:**
  - Downloads the dataset.
  - Crops faces from images.
  - Removes images without detectable faces if `remove_images` is set to `True`.

### 5. `embedding.py`
Handles the generation of feature embeddings and query trees for image retrieval.

**Key Functions:**
  - `build_embedding`: Converts an image into a feature embedding using a pre-trained CNN.
  - `build_query`: Builds a KD-tree for fast retrieval of similar images.

### 6. `upload_dataset.py`
Manages the uploading of processed datasets to Kaggle.

**Key Functions:** `upload_dataset`: Uploads or updates the dataset on Kaggle, using `kagglehub`.

### 7. `visualize.py`
Provides visualization tools for inspecting images and bounding boxes.

**Key Functions:** `plot_face_with_bounding_box`: Plots an image from the dataset with its bounding box.

## Prerequisites

You should install or update the following packages before running the scripts:
```bash
pip install -r requirements.txt
```

## Data Preparation
1. Download the CelebA dataset from [this link](https://www.kaggle.com/datasets/daoxuantan/my-celeba) and extract it.
2. Place the extracted images into the following directory:
   ```
   ~/image-processing-project/data/
   ```
   Create the folder structure if it does not exist.

## Usage

### 1. Download and Preprocess Data
Run the `pipeline.py` to download the dataset and crop faces:
```bash
python pipeline.py
```

### 2. Generate Embeddings
Generate feature embeddings using `embedding.py`:
```bash
python embedding.py
```

### 3. Visualize Data
Visualize an image with its bounding box using `visualize.py`:
```bash
python visualize.py
```

### 4. Upload Dataset
Upload the processed dataset to Kaggle using `upload_dataset.py`:
```bash
python upload_dataset.py
```



## Notes
- Ensure that you have the Haar cascade XML file (`haarcascade_frontalface_default.xml`) in the same directory as `processing.py`.
- If using KaggleHub for the first time, configure it with your Kaggle credentials.

## Contributions
Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

