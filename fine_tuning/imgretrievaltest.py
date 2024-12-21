import torch
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KDTree

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
ATTRIBUTES_PATH = os.path.join(PARENT_DIRNAME, "data/list_attr_celeba.csv")

def compute_embedding(model, dataloader, device):
    """
    Computes the embeddings for the dataset.

    Args:
        - model: Model to compute the embeddings.
        - dataloader: DataLoader for the dataset.
        - device: Device to run the model.

    Returns:
        - embeddings: Numpy array of embeddings.
        - filenames: List of filenames corresponding to the embeddings.
    """
    model.eval()
    embeddings = []
    filenames = []
    with torch.no_grad():
        for batch in dataloader:
            images, batch_filenames = batch
            images = images.to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            filenames.extend(batch_filenames)
    return np.vstack(embeddings), filenames

def evaluate_with_attributes(
    query_embeddings, gallery_embeddings,
    query_filenames, gallery_filenames, attributes_df, k=5
):
    """
    Evaluate retrieval performance using attributes similarity with KDTree.

    Args:
        query_embeddings (np.ndarray): Embeddings for the query images.
        gallery_embeddings (np.ndarray): Embeddings for the gallery images.
        query_filenames (list): Filenames for query images.
        gallery_filenames (list): Filenames for gallery images.
        attributes_df (pd.DataFrame): DataFrame containing attributes for each image.
        k (int): Number of top similar images to consider for evaluation.

    Returns:
        dict: Evaluation metrics including Top-K Attribute Accuracy and Mean Attribute Distance.
    """
    # Build KDTree for gallery embeddings
    tree = KDTree(gallery_embeddings)

    top_k_attribute_accuracies = []
    mean_attribute_distances = []

    for i, query_emb in enumerate(query_embeddings):
        # Query the KDTree for the top-K nearest neighbors
        _, indices = tree.query(query_emb.reshape(1, -1), k=k)
        top_k_indices = indices[0]

        # Retrieve attributes for query and top-K gallery images
        query_attributes = attributes_df[attributes_df['image_id'] == query_filenames[i]].iloc[:, 1:].values.flatten()
        top_k_attributes = attributes_df[attributes_df['image_id'].isin([gallery_filenames[j] for j in top_k_indices])]

        # Compute Attribute Accuracy and Distance
        attribute_accuracies = []
        attribute_distances = []
        total_attributes = len(query_attributes)
        for _, row in top_k_attributes.iterrows():
            gallery_attributes = row.iloc[1:].values
            match_count = np.sum(query_attributes == gallery_attributes)
            accuracy = match_count / total_attributes
            distance = np.sum(np.abs(query_attributes - gallery_attributes)) / total_attributes

            attribute_accuracies.append(accuracy)
            attribute_distances.append(distance)

        # Average over Top-K
        top_k_attribute_accuracies.append(np.mean(attribute_accuracies))
        mean_attribute_distances.append(np.mean(attribute_distances))

    return {
        "Top-K Attribute Accuracy": np.mean(top_k_attribute_accuracies),
        "Mean Attribute Distance": np.mean(mean_attribute_distances),
    }

# Example usage:
def run_evaluation_pipeline_with_attributes(
    query_loader, gallery_loader, model, device, attributes_path=ATTRIBUTES_PATH, k=5
):
    """
    Compute embeddings and evaluate retrieval performance using attributes with KDTree.

    Args:
        query_loader (DataLoader): DataLoader for the query images.
        gallery_loader (DataLoader): DataLoader for the gallery images.
        model (torch.nn.Module): The trained model for embedding generation.
        device (torch.device): Device for computation (CPU/GPU).
        attributes_path (str): Path to the attributes CSV file.
        k (int): Number of top similar images to consider.

    Returns:
        dict: Evaluation metrics.
    """
    model.eval()

    query_embeddings, query_filenames = compute_embedding(model, query_loader, device)
    gallery_embeddings, gallery_filenames = compute_embedding(model, gallery_loader, device)

    attributes_df = pd.read_csv(attributes_path)

    print("Evaluating retrieval performance using attributes and KDTree...")
    metrics = evaluate_with_attributes(
        query_embeddings, gallery_embeddings,
        query_filenames, gallery_filenames,
        attributes_df, k=k
    )

    print(f"Top-{k} Attribute Accuracy: {metrics['Top-K Attribute Accuracy']:.4f}")
    print(f"Mean Attribute Distance: {metrics['Mean Attribute Distance']:.4f}")
    return metrics
