import matplotlib.pyplot as plt
import matplotlib.image as mpimg # `mpimg.imread` is used to read images
import os
from sklearn.neighbors import KDTree
import torch
from torchvision.transforms import transforms
from PIL import Image
from fine_tuning.retrieval_models import RetrievalModel
import numpy as np

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
STORAGE_DATA_DIRNAME = os.path.join(PARENT_DIRNAME, "fine_tuning/data_for_fine_tuning")
MODEL_DIR = os.path.join(PARENT_DIRNAME, "fine_tuning/models")

def plot_retrieved_images(query_image_path, gallery_image_paths, distances, top_k):
    """
    Plot the query image on the leftmost side and the top-K retrieved images on the right.
    The query image is centered vertically relative to the retrieved images.

    Args:
        - query_image_path: Path to the query image.
        - gallery_image_paths: List of gallery image paths.
        - distances: List of distances for the retrieved images.
        - top_k: Number of retrieved images to display.
    """
    if not gallery_image_paths:
        print("No images to display. Gallery image paths are empty.")
        return

    # Adjust the number of gallery images to match top_k
    gallery_image_paths = gallery_image_paths[:top_k]
    distances = distances[:top_k]

    # Calculate the number of rows and columns for gallery images
    num_columns = 3
    num_rows = (top_k + num_columns - 1) // num_columns

    # Create a figure for the plot
    fig, axs = plt.subplots(num_rows, num_columns + 1, figsize=(20, num_rows * 5), gridspec_kw={"width_ratios": [0.8] + [1] * num_columns})
    fig.subplots_adjust(wspace=0.4, hspace=0.4)

    # Plot the query image on the leftmost column, vertically centered
    query_img = mpimg.imread(query_image_path)
    query_row_start = (num_rows - 1) // 2
    query_row_end = query_row_start + 1

    for row in range(num_rows):
        if query_row_start <= row < query_row_end:
            axs[row, 0].imshow(query_img)
            axs[row, 0].set_title("Query Image", fontsize=14, fontweight="bold", color="darkblue")
            axs[row, 0].axis("off")
        else:
            axs[row, 0].axis("off")

    # Plot the retrieved images on the right
    for idx, gallery_image_path in enumerate(gallery_image_paths):
        row, col = divmod(idx, num_columns)
        ax = axs[row, col + 1]  # Fill right columns with retrieved images
        gallery_img = mpimg.imread(gallery_image_path)
        ax.imshow(gallery_img)
        ax.set_title(f"Rank {idx + 1}\nDistance: {distances[idx]:.2f}", fontsize=10, color="green")
        ax.axis("off")

    # Hide unused subplots for retrieved images
    for idx in range(len(gallery_image_paths), num_rows * num_columns):
        row, col = divmod(idx, num_columns)
        axs[row, col + 1].axis("off")

    plt.show()

def query_and_plot_images(
    query_image_path, model, top_k=5, device="cuda"
):
    """
    Query an image and plot the top-K most similar images from the dataset.

    Args:
        - query_image_path: Path to the query image.
        - model: The model used to compute embeddings.
        - top_k: Number of top similar images to retrieve.
        - device: Device to run the model.
    """
    print("Loading precomputed embeddings and labels...")

    full_embeddings_path = ""
    full_labels_path = ""

    if model == "mobilenet_v2":
        model = RetrievalModel(backbone="mobilenet_v2", embedding_dim=128).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "mobilenet_v2_identity.pth")))
        full_embeddings_path = os.path.join(STORAGE_DATA_DIRNAME, "full_embeddings_mobilenet.pth")
        full_labels_path = os.path.join(STORAGE_DATA_DIRNAME, "full_labels_mobilenet.pth")

    elif model == "resnet50":
        model = RetrievalModel(backbone="resnet50", embedding_dim=128).to(device)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet50_identity.pth")))
        full_embeddings_path = os.path.join(STORAGE_DATA_DIRNAME, "full_embeddings_resnet.pth")
        full_labels_path = os.path.join(STORAGE_DATA_DIRNAME, "full_labels_resnet.pth")

    full_embeddings = torch.load(full_embeddings_path)
    full_labels = torch.load(full_labels_path)

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((218, 218)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Compute the query embedding
    model.eval()
    with torch.no_grad():
        image = Image.open(query_image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        query_embedding = model(image).cpu().numpy()

    # Query the KDTree for the top-K nearest neighbors
    kdtree = KDTree(full_embeddings)
    distances, indices = kdtree.query(query_embedding.reshape(1, -1), k=top_k)
    top_k_indices = indices[0]

    # Construct the paths for the top-K retrieved images
    gallery_image_paths = [full_labels[i] for i in top_k_indices]

    # Plot the query image and retrieved images
    plot_retrieved_images(query_image_path, gallery_image_paths, distances[0], top_k=top_k)

    return gallery_image_paths, distances[0]