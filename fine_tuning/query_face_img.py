import numpy as np
import torch
import os
from sklearn.neighbors import KDTree
from PIL import Image
from torchvision import transforms
from retrievalmodels import RetrievalModel
from plotretrievedimg import plot_retrieved_images

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba/")
STORAGE_DATA_DIRNAME = os.path.join(PARENT_DIRNAME, "fine_tuning/data_for_fine_tuning")
MODEL_DIR = os.path.join(PARENT_DIRNAME, "fine_tuning/models")
FULL_EMBEDDINGS_PATH = os.path.join(STORAGE_DATA_DIRNAME, "full_embeddings.pth")
FULL_LABELS_PATH = os.path.join(STORAGE_DATA_DIRNAME, "full_labels.pth")
IMAGE_SIZE = 218

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def compute_single_embedding(model, image_path, transformer=transform, device="cuda"):
    """
    Computes the embedding for a single image.

    Args:
        - model: The model used to compute embeddings.
        - image_path: Path to the query image.
        - transform: Transformation to apply to the image.
        - device: Device to run the model.

    Returns:
        - embedding: Embedding of the image.
    """
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        image = transformer(image).unsqueeze(0).to(device)
        embedding = model(image).cpu().numpy()
    return embedding

def retrieve_top_k(query_embedding, full_embeddings, full_labels, top_k):
    """
    Retrieve the top-K most similar images to the query embedding using KDTree.

    Args:
        - query_embedding: Embedding of the query image.
        - full_embeddings: Precomputed embeddings for the entire dataset.
        - full_labels: Labels corresponding to the embeddings.
        - top_k: Number of top similar images to retrieve.

    Returns:
        - top_k_indices: Indices of the top-K most similar images.
        - top_k_labels: Labels of the top-K most similar images.
    """
    kdtree = KDTree(full_embeddings)
    _, indices = kdtree.query(query_embedding.reshape(1, -1), k=top_k)
    top_k_indices = indices[0]
    top_k_labels = [full_labels[i] for i in top_k_indices]
    return top_k_indices, top_k_labels

def query_and_plot_images(
    query_image_path, model="mobilenet_v2",
    image_dir=IMAGE_DIR, 
    top_k=5,
    full_embeddings_path=FULL_EMBEDDINGS_PATH,
    full_labels_path=FULL_LABELS_PATH, 
    device="cuda"
):
    """
    Query an image and plot the top-K most similar images from the dataset.

    Args:
        - model: The model used to compute embeddings.
        - query_image_path: Path to the query image.
        - full_embeddings_path: Path to the file containing precomputed embeddings.
        - full_labels_path: Path to the file containing labels for embeddings.
        - image_dir: Directory containing the dataset images.
        - top_k: Number of top similar images to retrieve.
        - device: Device to run the model.
    """
    if model == "mobilenet_v2":
        model = RetrievalModel(backbone="mobilenet_v2", embedding_dim=128).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenet_v2_identity.pth")))

    elif model == "resnet50":
        model = RetrievalModel(backbone="resnet50", embedding_dim=256).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet50_identity.pth")))

    print("Loading precomputed embeddings and labels...")
    full_embeddings = torch.load(full_embeddings_path)
    full_labels = torch.load(full_labels_path)

    print("Computing embedding for the query image...")
    query_embedding = compute_single_embedding(model, query_image_path, transform, device)

    print("Retrieving top-K similar images...")
    _, top_k_labels = retrieve_top_k(query_embedding, full_embeddings, full_labels, top_k)

    # Construct the paths for the top-K retrieved images
    gallery_image_paths = [os.path.join(image_dir, label) for label in top_k_labels]

    print("Plotting the query image and retrieved images...")
    plot_retrieved_images(query_image_path, gallery_image_paths, top_k=top_k)