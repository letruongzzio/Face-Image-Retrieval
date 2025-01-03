# Fine Tuning - Image Processing and Face Retrieval Project

## Overview
This section demonstrates a pipeline for face image retrieval using the CelebA dataset. It includes modules for embedding computation, model fine-tuning, triplet data generation, and retrieval evaluation. Example notebooks showcase training and inference processes, and pre-trained embeddings and models are provided for user convenience.

## Prerequisites

You should install or update the following packages before running the scripts:
```bash
pip install -r requirements.txt
```

## Getting Started
### Data Preparation
1. Download the CelebA dataset from [this link](https://www.kaggle.com/datasets/daoxuantan/my-celeba) and extract it.
2. Place the extracted images into the following directory:
   ```
   ~/image-processing-project/data/
   ```
   Create the folder structure if it does not exist.

3. Pre-trained embeddings and labels for the dataset are available [here](https://drive.google.com/drive/u/3/folders/1ut9KtKUNWa3krGMpkcFLKpsi_yxESeNo). Download and place them into:
   ```
   ~/image-processing-project/fine_tuning/data_for_fine_tuning/
   ```

### Directory Structure
After setup, your project directory should look like this:
```
image-processing-project/
├── data/
├── fine_tuning/
│   ├── data_for_fine_tuning/
│   │   ├── full_embeddings_mobilenet.pth
│   │   ├── full_embeddings_resnet.pth
│   │   ├── ...
│   ├── models/
│       ├── mobilenet_v2_identity.pth
│       ├── resnet50_identity.pth
│   ├── finetune_data_processing_example.ipynb
│   ├── finetune_models_example.ipynb
│   ├── finetune_query_evaluate_example.ipynb
│   ├── *.py
```

## Scripts and Notebooks

### Python Scripts

#### 1. `compute_embedding_celebA.py`
- **Purpose**: Computes embeddings for all images in the CelebA dataset using a specified model.
- **Key Functionality**:
  - Loads images, preprocesses them, and computes embeddings in batches.
  - Handles missing or corrupt images gracefully.
- **Usage**: Modify the script to use your model, set the device (e.g., `cuda` or `cpu`), and run it to generate embeddings for retrieval tasks.

#### 2. `dropout.py`
- **Purpose**: Applies random dropout to image tensors as a regularization technique.
- **Key Functionality**: Implements PyTorch's `nn.Dropout` for image augmentation during training.
- **Usage**: Can be imported into your training pipeline for enhanced model generalization.

#### 3. `imgretrievaltest.py`
- **Purpose**: Tests retrieval accuracy using pre-computed embeddings and KDTree for nearest neighbor search.
- **Key Functionality**:
  - Computes query and gallery embeddings.
  - Uses KDTree for efficient similarity searches.
  - Calculates Precision@K, Recall@K, F1, and Accuracy.
- **Usage**: Run the script to evaluate the retrieval performance of your model.

#### 4. `modelfinetuning.py`
- **Purpose**: Fine-tunes pre-trained models using triplet loss.
- **Key Functionality**:
  - Supports MobileNetV2 and ResNet50 backbones.
  - Uses triplet datasets for optimizing embedding spaces.
- **Usage**: Fine-tune the model to improve retrieval performance.

#### 5. `query_face_img.py`
- **Purpose**: Queries an image and retrieves the most similar images.
- **Key Functionality**:
  - Loads pre-trained embeddings and models.
  - Computes query embeddings and retrieves top-K matches using KDTree.
  - Visualizes results with the query image and retrieved images.
- **Usage**: Adjust the script with the desired query image and model.

#### 7. `retrievalmodels.py`
- **Purpose**: Defines the neural network models for embedding generation.
- **Key Functionality**:
  - Supports MobileNetV2 and ResNet50 as backbones.
  - Projects features into a low-dimensional embedding space.
- **Usage**: Import this script to define your model architecture.

#### 8. `tripletmethod.py`
- **Purpose**: Provides triplet loss and dataset classes for training models.
- **Key Functionality**:
  - Implements the `TripletDataset` and `QueryDataset` for efficient data handling.
  - Defines triplet loss for optimizing embedding spaces.
- **Usage**: Essential for training pipelines that use triplet loss.

#### 9. `tripletprocess.py`
- **Purpose**: Generates triplets for training and testing using identity labels.
- **Key Functionality**:
  - Splits identities into train/test sets.
  - Creates anchor-positive-negative triplets for training.
  - Generates query and gallery pairs for testing.
- **Usage**: Use this script to prepare datasets for fine-tuning and evaluation.

### Notebooks

#### 1. `finetune_data_processing_example.ipynb`
- **Purpose**: Demonstrates how to process data for training and testing.
- **Key Functionality**: Walks through triplet generation and data preparation steps.
- **Usage**: Reference this notebook to set up your data pipeline.

#### 2. `finetune_models_example.ipynb`
- **Purpose**: Provides an example of model fine-tuning.
- **Key Functionality**: Demonstrates model training using triplet loss.
- **Usage**: Modify and execute this notebook for your fine-tuning tasks.

#### 3. `finetune_query_evaluate_example.ipynb`
- **Purpose**: Shows query-based retrieval evaluation.
- **Key Functionality**:
  - Loads pre-trained embeddings and models.
  - Queries images and evaluates retrieval accuracy.
- **Usage**: Use this notebook to understand the retrieval process and measure performance.

## Notes
- **Data Format**: Ensure that your dataset and embeddings follow the specified directory structure.
- **Custom Pipeline**: This project does not include a complete pipeline. Use the provided scripts and notebooks as building blocks for your own workflows.
- **Model Selection**: Choose the appropriate model for your task based on the desired trade-offs between speed and accuracy.

## Contributions
Contributions are welcome! Please feel free to submit issues or pull requests for improvements.
