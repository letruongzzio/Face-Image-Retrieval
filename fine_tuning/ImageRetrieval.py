import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_embedding(model, dataloader, device):
    """
    Computes the embeddings for the dataset.

    Args:
        - model: Model to compute the embeddings.
        - dataloader: DataLoader for the dataset.
        - device: Device to run the model.

    Returns:
        - embeddings: Numpy array of embeddings.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            emb, _ = model(images)  # Forward pass to get embeddings
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def retrieve_similar_images(query_embedding, gallery_embeddings, attributes=None, query_attributes=None, top_k=5, threshold=0.8):
    """
    Retrieve similar images from the gallery using cosine similarity with optional attribute filtering.

    Args:
        - query_embedding: Query embedding (1D array).
        - gallery_embeddings: Gallery embeddings (2D array).
        - attributes: Attributes for gallery images (optional).
        - query_attributes: Attributes for the query image (optional).
        - top_k: Number of similar images to retrieve.
        - threshold: Threshold for attribute similarity filtering.

    Returns:
        - filtered_indices: Indices of the top-K similar images after threshold filtering.
    """
    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, gallery_embeddings)
    top_k_indices = similarities.argsort(axis=1)[:, -top_k:]

    # Apply threshold for attribute filtering (optional)
    if query_attributes is not None and attributes is not None:
        filtered_indices = []
        for idx in top_k_indices[0]:
            attr_similarity = np.mean(attributes[idx] == query_attributes)
            if attr_similarity >= threshold:  # Only accept if above threshold
                filtered_indices.append(idx)
        return filtered_indices[:top_k]  # Return filtered top-K results
    
    return top_k_indices[0]  # Default top-K without filtering


def evaluate_test_set(model, query_loader, gallery_loader, device, top_k=5):
    """
    Evaluate the model using Precision@K and Recall@K.

    Args:
        - model: Model to evaluate.
        - query_loader: DataLoader for the query set.
        - gallery_loader: DataLoader for the gallery set.
        - device: Device to run the model.
        - top_k: Number of similar images to retrieve.

    Returns:
        - avg_precision: Average Precision@K.
        - avg_recall: Average Recall@K.
    """
    # Compute embeddings
    print("Computing embeddings for the query set...")
    query_embeddings = compute_embedding(model, query_loader, device)
    print("Computing embeddings for the gallery set...")
    gallery_embeddings = compute_embedding(model, gallery_loader, device)

    precision_at_k = []
    recall_at_k = []

    # Calculate Precision and Recall for each query
    for i, query_emb in enumerate(query_embeddings):
        query_emb = query_emb.reshape(1, -1)
        top_k_indices = retrieve_similar_images(query_emb, gallery_embeddings, top_k=top_k)

        # Placeholder: Relevant check based on index
        relevant = sum(1 for idx in top_k_indices if idx == i)  # Simplified for demo purposes
        total_relevant = 1  # Placeholder

        # Compute Precision and Recall
        precision = relevant / top_k
        recall = relevant / total_relevant

        precision_at_k.append(precision)
        recall_at_k.append(recall)

    avg_precision = np.mean(precision_at_k)
    avg_recall = np.mean(recall_at_k)

    print(f"Precision@{top_k}: {avg_precision:.4f}, Recall@{top_k}: {avg_recall:.4f}")
    return avg_precision, avg_recall