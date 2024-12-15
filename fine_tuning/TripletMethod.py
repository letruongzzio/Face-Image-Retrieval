import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class TripletDataset(Dataset):
    """
    Custom Dataset for loading triplet data.
    """
    def __init__(self, image_dir, train_triplets, attributes=None, transform=None):
        """
        Args:
            - image_dir: Path to the image directory.
            - train_triplets: List of (anchor, positive, negative) triplets.
            - attributes: Dictionary of attributes for each image.
            - transform: PyTorch transformations.
        """
        self.image_dir = image_dir
        self.train_triplets = train_triplets
        self.attributes = attributes
        self.transform = transform

    def __len__(self):
        """
        Returns the number of triplets.
        """
        return len(self.train_triplets)

    def __getitem__(self, idx):
        """
        Returns the triplet (anchor, positive, negative) and their attributes.
        """
        anchor_path, positive_path, negative_path = self.train_triplets[idx]

        # Check existence of files
        if not all(os.path.exists(os.path.join(self.image_dir, p)) for p in [anchor_path, positive_path, negative_path]):
            print(f"Warning: One or more images in triplet {idx} not found. Skipping.")
            return None

        anchor = Image.open(os.path.join(self.image_dir, anchor_path)).convert("RGB")
        positive = Image.open(os.path.join(self.image_dir, positive_path)).convert("RGB")
        negative = Image.open(os.path.join(self.image_dir, negative_path)).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        if self.attributes:
            anchor_attr = torch.tensor(self.attributes[anchor_path], dtype=torch.float32)
            positive_attr = torch.tensor(self.attributes[positive_path], dtype=torch.float32)
            negative_attr = torch.tensor(self.attributes[negative_path], dtype=torch.float32)
            return anchor, positive, negative, anchor_attr, positive_attr, negative_attr

        return anchor, positive, negative


class QueryDataset(Dataset):
    """
    Custom Dataset for loading query data.
    """
    def __init__(self, image_dir, query_triplets, transform=None):
        """
        Dataset for query images.

        Args:
            - image_dir: Path to the image directory.
            - query_triplets: List of (query_image,).
            - transform: Transformations to apply to images.
        """
        self.image_dir = image_dir
        self.query_triplets = query_triplets
        self.transform = transform

    def __len__(self):
        return len(self.query_triplets)

    def __getitem__(self, idx):
        query_image_path = self.query_triplets[idx][0]
        query_image = Image.open(os.path.join(self.image_dir, query_image_path)).convert("RGB")
        if self.transform:
            query_image = self.transform(query_image)
        return query_image, query_image_path

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Computes the triplet loss.

    Args:
        - anchor: anchor embeddings
        - positive: positive embeddings
        - negative: negative embeddings
        - margin: margin value

    Returns:
        - triplet loss
    """
    pos_dist = F.pairwise_distance(anchor, positive, p=2)
    neg_dist = F.pairwise_distance(anchor, negative, p=2)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()