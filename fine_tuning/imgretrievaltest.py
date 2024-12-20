import torch
import numpy as np
import os
from sklearn.neighbors import KDTree

PARENT_DIRNAME = os.path.expanduser("~/image-processing-project/")
IMAGE_DIR = os.path.join(PARENT_DIRNAME, "data/img_align_celeba/")

def compute_embedding(model, dataloader, device):
    """
    Computes the embeddings for the dataset.

    Args:
        - model: Model to compute the embeddings.
        - dataloader: DataLoader for the dataset.
        - device: Device to run the model.

    Returns:
        - embeddings: Numpy array of embeddings.
        - labels: List of labels corresponding to embeddings.
    """
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            images, batch_labels = batch
            images = images.to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            labels.extend(batch_labels)
    return np.vstack(embeddings), labels

def build_kdtree(embeddings):
    """
    Build a KDTree for fast nearest neighbor search.

    Args:
        - embeddings: Numpy array of embeddings.

    Returns:
        - tree: KDTree object.
    """
    return KDTree(embeddings)

def retrieve_similar_images_kdtree(query_embedding, kdtree, gallery_labels, top_k=5):
    """
    Retrieve similar images using KDTree.

    Args:
        - query_embedding: Embedding for the query image (1D array).
        - kdtree: Pre-built KDTree object.
        - gallery_labels: Labels of the gallery embeddings.
        - top_k: Number of similar images to retrieve.

    Returns:
        - top_k_indices: Indices of the top-K similar images.
        - top_k_labels: Labels of the top-K similar images.
    """
    _, indices = kdtree.query(query_embedding.reshape(1, -1), k=top_k)
    top_k_indices = indices[0]
    top_k_labels = [gallery_labels[i] for i in top_k_indices]
    return top_k_indices, top_k_labels

def calculate_accuracy(query_embeddings, gallery_embeddings, query_labels, gallery_labels):
    """
    Calculate accuracy based on the closest match in embeddings.

    Args:
        - query_embeddings: Numpy array of query embeddings.
        - gallery_embeddings: Numpy array of gallery embeddings.
        - query_labels: List of query labels.
        - gallery_labels: List of gallery labels.

    Returns:
        - accuracy: Overall accuracy based on the closest match.
    """
    correct = 0
    total = len(query_embeddings)

    for query_emb, query_label in zip(query_embeddings, query_labels):
        # Calculate distances between the query embedding and gallery embeddings
        distances = np.linalg.norm(gallery_embeddings - query_emb, axis=1)
        closest_index = np.argmin(distances)
        predicted_label = gallery_labels[closest_index]

        if predicted_label == query_label:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_test_set_kdtree(model, query_loader, gallery_loader, device, top_k=5):
    """
    Evaluate the model using KDTree for Precision@K, Recall@K, F1@K, and Accuracy.

    Args:
        - model: Model to evaluate.
        - query_loader: DataLoader for the query set.
        - gallery_loader: DataLoader for the gallery set.
        - device: Device to run the model.
        - top_k: Number of similar images to retrieve.

    Returns:
        - avg_precision: Average Precision@K.
        - avg_recall: Average Recall@K.
        - avg_f1: Average F1@K.
        - accuracy: Overall accuracy.
    """
    print("Step 1: Computing test-query and test-gallery embeddings...")
    query_embeddings, query_labels = compute_embedding(model, query_loader, device)
    gallery_embeddings, gallery_labels = compute_embedding(model, gallery_loader, device)

    print("Step 2: Building KDTree for gallery embeddings...")
    kdtree = build_kdtree(gallery_embeddings)

    precision_at_k = []
    recall_at_k = []

    print("Step 3: Evaluating Precision@K and Recall@K...")
    for query_emb, query_label in zip(query_embeddings, query_labels):
        _, top_k_labels = retrieve_similar_images_kdtree(query_emb, kdtree, gallery_labels, top_k=top_k)

        relevant = sum(1 for label in top_k_labels if label == query_label)
        total_relevant = query_labels.count(query_label)

        precision = relevant / top_k
        recall = relevant / total_relevant if total_relevant > 0 else 0

        precision_at_k.append(precision)
        recall_at_k.append(recall)

    avg_precision = np.mean(precision_at_k)
    avg_recall = np.mean(recall_at_k)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    print("Step 4: Calculating Accuracy...")
    accuracy = calculate_accuracy(query_embeddings, gallery_embeddings, query_labels, gallery_labels)

    print(f"Precision@{top_k}: {avg_precision:.4f}")
    print(f"Recall@{top_k}: {avg_recall:.4f}")
    print(f"F1@{top_k}: {avg_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")


def run_testing_pipeline_kdtree(model, query_loader, gallery_loader, device, top_k=5):
    """
    Run the full pipeline: compute embeddings, build KDTree, evaluate the model.

    Args:
        - model: Model to evaluate.
        - query_loader: DataLoader for the query set.
        - gallery_loader: DataLoader for the gallery set.
        - device: Device to run the model.
        - top_k: Number of similar images to retrieve.

    Returns:
        - avg_precision: Average Precision@K.
        - avg_recall: Average Recall@K.
        - avg_f1: Average F1@K.
        - accuracy: Overall accuracy.
    """
    print("Evaluating with KDTree...")
    evaluate_test_set_kdtree(model, query_loader, gallery_loader, device, top_k=top_k)